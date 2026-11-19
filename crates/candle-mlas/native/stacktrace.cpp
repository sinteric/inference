// The standalone upstream quantization objects use this error-path helper.
// Provide a real Linux stack trace without linking the ONNX inference runtime.
#include <execinfo.h>
#include <cstdlib>
#include <string>
#include <vector>

namespace onnxruntime {
std::vector<std::string> GetStackTrace() {
    void* frames[64];
    const int count=backtrace(frames,64);
    char** names=backtrace_symbols(frames,count);
    if (!names) return {"Stack trace unavailable"};
    std::vector<std::string> out;
    out.reserve(count);
    for (int i=0;i<count;++i) out.emplace_back(names[i]);
    std::free(names);
    return out;
}
}
