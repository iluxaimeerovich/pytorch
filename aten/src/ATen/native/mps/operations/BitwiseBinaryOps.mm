#include <ATen/mps/MPSStream.h>
#include <ATen/native/Resize.h>
#include <fmt/format.h>
#include <torch/library.h>

namespace {
static const char* BITWISE_OPS_KERNELS = R"METAL(

kernel void bitwise_and_tensor(constant uint& length [[buffer(0)]],
                         device {0}  *out [[buffer(1)]],
                         device {1}  *a [[buffer(2)]],
                         device {2}  *b [[buffer(3)]],
                         uint offset [[thread_position_in_grid]]) {{
  if (offset >= length) {{
    return;
  }}
  out[offset] = a[offset] & b [offset];
}}

kernel void bitwise_and_scalar(constant uint& length [[buffer(0)]],
                         device {0}  *out [[buffer(1)]],
                         device {1}  *a [[buffer(2)]],
                         constant {2}  &b [[buffer(3)]],
                         uint offset [[thread_position_in_grid]]) {{
  if (offset >= length) {{
    return;
  }}
  out[offset] = a[offset] & b;
}}


)METAL";

static std::unordered_map<c10::ScalarType, std::string> scalar_to_metal_type = {
  {c10::ScalarType::Long, "long"},
  {c10::ScalarType::Int, "int"},
  {c10::ScalarType::Short, "short"},
  {c10::ScalarType::Byte, "char"},
  {c10::ScalarType::Char, "char"},
  {c10::ScalarType::Bool, "char"},
};

const std::string& getMetalType(const c10::ScalarType& t) {
  auto it = scalar_to_metal_type.find(t);
  TORCH_CHECK(it != scalar_to_metal_type.end(), "Unsupported type ", t);
  return it->second;
}

const std::string& getMetalType(const at::Tensor& t) {
  return getMetalType(t.scalar_type());
}

const std::string& getMetalType(const c10::Scalar& s) {
  return getMetalType(s.type());
}


static id<MTLLibrary> compileBitwiseOpsLibrary(id<MTLDevice> device, const std::string& t1, const std::string& t2, const std::string& t3) {
  auto key = t1 + t2 + t3;
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError *error = nil;
  auto rc  = [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(BITWISE_OPS_KERNELS, t1, t2, t3).c_str()]
                                  options:nil
                                    error:&error];
 TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
 libMap[key] = rc;
 return rc;
}


static id<MTLComputePipelineState> getCPLState(id<MTLDevice> device, const std::string& t1, const std::string& t2, const std::string& t3, const std::string& fname) {
  auto key = t1 + t2 + t3 + fname;
 static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
 auto it = cplMap.find(key);
 if (it != cplMap.end()) {
    return it->second;
 }
 NSError *error = nil;
 auto library = compileBitwiseOpsLibrary(device, t1, t2, t3);
 id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
 TORCH_CHECK(func != nil, "Can't get function", fname);
 id<MTLComputePipelineState> rc = [device newComputePipelineStateWithFunction:func error:&error];
 TORCH_CHECK(rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
 cplMap[key]  = rc;
 return rc;
}

void handle_tensor_tensor_binary_op(const at::Tensor& self, const at::Tensor& other, at::Tensor& output, const std::string& kernel_name) {
  using namespace at::mps;
  MPSStream* stream = getCurrentMPSStream();
  id<MTLComputePipelineState> cplState = getCPLState(MPSDevice::getInstance()->device(),
                                                     getMetalType(output),
                                                     getMetalType(self),
                                                     getMetalType(other),
                                                     kernel_name);
  uint32_t length = output.numel();
  dispatch_sync(stream->queue(), ^(){
    id<MTLCommandBuffer> buffer = stream->commandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = [buffer computeCommandEncoder];

    id<MTLBuffer> outBuf = __builtin_bit_cast(id<MTLBuffer>, output.storage().data());
    id<MTLBuffer> selfBuf = __builtin_bit_cast(id<MTLBuffer>, self.storage().data());
    id<MTLBuffer> otherBuf = __builtin_bit_cast(id<MTLBuffer>, other.storage().data());

    [commandEncoder pushDebugGroup:[NSString stringWithFormat:@"Dispatch %s kernel", kernel_name.c_str()]];
    [commandEncoder setComputePipelineState:cplState];
    [commandEncoder setBytes:&length length:sizeof(length) atIndex:0];
    [commandEncoder setBuffer:outBuf offset:output.storage_offset()*output.itemsize() atIndex:1];
    [commandEncoder setBuffer:selfBuf offset:self.storage_offset()*self.itemsize()  atIndex:2];
    [commandEncoder setBuffer:otherBuf offset:other.storage_offset()*other.itemsize() atIndex:3];
    [commandEncoder dispatchThreadgroups:MTLSizeMake((length + 511) / 512, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    [commandEncoder endEncoding];
  });
}

void handle_tensor_scalar_binary_op(const at::Tensor& self, const at::Scalar& other, at::Tensor& output, const std::string& kernel_name) {
  using namespace at::mps;
  MPSStream* stream = getCurrentMPSStream();
  id<MTLComputePipelineState> cplState = getCPLState(MPSDevice::getInstance()->device(),
                                                     getMetalType(output),
                                                     getMetalType(self),
                                                     getMetalType(other),
                                                     kernel_name);
  uint64_t sval = other.to<int64_t>();
  uint32_t length = output.numel();
  dispatch_sync(stream->queue(), ^(){
    id<MTLCommandBuffer> buffer = stream->commandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = [buffer computeCommandEncoder];

    id<MTLBuffer> outBuf = __builtin_bit_cast(id<MTLBuffer>, output.storage().data());
    id<MTLBuffer> selfBuf = __builtin_bit_cast(id<MTLBuffer>, self.storage().data());

    [commandEncoder pushDebugGroup:[NSString stringWithFormat:@"Dispatch %s kernel", kernel_name.c_str()]];
    [commandEncoder setComputePipelineState:cplState];
    [commandEncoder setBytes:&length length:sizeof(length) atIndex:0];
    [commandEncoder setBuffer:outBuf offset:output.storage_offset()*output.itemsize() atIndex:1];
    [commandEncoder setBuffer:selfBuf offset:self.storage_offset()*self.itemsize()  atIndex:2];
    [commandEncoder setBytes:&sval length:sizeof(sval) atIndex:3];
    [commandEncoder dispatchThreadgroups:MTLSizeMake((length + 511) / 512, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    [commandEncoder endEncoding];
  });
}

at::Tensor& bitwise_and_out_mps (const at::Tensor& self, const at::Tensor& other, at::Tensor& output) {
  using namespace at::mps;
  const bool is_self_scalar = self.dim() == 0;
  const bool is_other_scalar = other.dim() == 0;
  at::native::resize_output(output, self.sizes());
  if (is_other_scalar && is_self_scalar) {
    output.fill_(c10::Scalar(self.item<int64_t>() & other.item<int64_t>()));
  } else if (is_other_scalar) {
    handle_tensor_scalar_binary_op(self, other.item(), output, "bitwise_and_scalar_long");
  } else if (is_self_scalar) {
    handle_tensor_scalar_binary_op(other, self.item(), output, "bitwise_and_scalar_long");
  } else {
    handle_tensor_tensor_binary_op(self, other, output, "bitwise_and_tensor");
  }
  return output;
}

TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("bitwise_and.Tensor_out", bitwise_and_out_mps);
}

} // anonymous namespace
