/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "tensorrt.hpp"

#include <onnxruntime_cxx_api.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <sstream>

using namespace std;

namespace TensorRT {

static std::string format_shape(const Ort::ShapeInferContext::Ints &shape) {

std::stringstream ss;
for (size_t i = 0; i < shape.size(); ++i) {
    if (i + 1 < shape.size())
        ss << shape[i] << " x ";
    else
        ss << shape[i];
}
return ss.str();
}

static std::vector<uint8_t> load_file(const std::string &file) {
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, std::ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, std::ios::beg);
    data.resize(length);

    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

static const char *data_type_string(ONNXTensorElementDataType dt) {
  switch (dt) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "Float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "Float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "Int32";
    // case nvinfer1::DataType::kUINT8: return "UInt8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "Int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "BOOL";
    default:
      return "Unknow";
  }
}

template <typename _T>
static void destroy_pointer(_T *ptr) {
  if (ptr) delete ptr;
}



class EngineImplement : public TensorRT::Engine {
public:
    EngineImplement(const std::string& model_path);
    virtual ~EngineImplement();
    bool forward(const std::vector<const void*>& bindings, void* stream = nullptr, void* input_consum_event = nullptr) override;
    int index(const std::string &name) override;
    std::vector<int> run_dims(const std::string &name)  override;
    std::vector<int> run_dims(int ibinding) override;
    std::vector<int> static_dims(const std::string &name) override;
    std::vector<int> static_dims(int ibinding) override;
    int numel(const std::string &name) override;
    int numel(int ibinding)  override;
    int num_bindings() override;
    bool is_input(int ibinding)  override;
    bool set_run_dims(const std::string &name, const std::vector<int> &dims) override;
    bool set_run_dims(int ibinding, const std::vector<int> &dims) override;
    DType dtype(const std::string &name) override;
    DType dtype(int ibinding) override;
    bool has_dynamic_dim() override;
    void print(const char *name = "TensorRT-Engine") override;

private:
    void initializeIO();
    size_t numel(const std::vector<int64_t>& dims);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo allocator_info_;

    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_;
    std::vector<std::vector<int64_t>> output_node_dims_;
};

size_t EngineImplement::numel(const std::vector<int64_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
}

EngineImplement::EngineImplement(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntime"),
      session_options_(),
      session_(nullptr),
      allocator_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault)) {
    
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    initializeIO();
}

int32_t EngineImplement::index(const std::string &name) {
    for(int32_t i = 0; i < input_node_names_.size(); i++) {
        if (name.compare(input_node_names_[i]) == 0) {
            return i;
        }
    }

    for (int32_t i = 0; i < output_node_names_.size(); i++) {
        if ( name.compare(output_node_names_[i]) == 0) {
            return input_node_names_.size()+i;
        }
    }
    return -1;
}

std::vector<int> EngineImplement::run_dims(const std::string &name) {
    int32_t idx = index(name);
    if (idx < 0) {
        return {};
    }
    return run_dims(idx);
}

std::vector<int> EngineImplement::run_dims(int ibinding) {
    if (ibinding < 0 || ibinding >= input_node_dims_.size()+output_node_dims_.size()) {
        return {};
    }
    std::vector<int> dims;
    if (ibinding < input_node_dims_.size()) {
        for (int i = 0; i < input_node_dims_[ibinding].size(); i++) {
            dims.push_back(input_node_dims_[ibinding][i]);
        }
    } else {
        for (int i = 0; i < output_node_dims_[ibinding-input_node_dims_.size()].size(); i++) {
            dims.push_back(output_node_dims_[ibinding-input_node_dims_.size()][i]);
        }
    }

    return dims;
}

std::vector<int> EngineImplement::static_dims(const std::string &name) {
    int32_t idx = index(name);
    if (idx < 0) {
        return {};
    }
    return static_dims(idx);
}

std::vector<int> EngineImplement::static_dims(int ibinding) {
    return run_dims(ibinding);
}

int EngineImplement::numel(const std::string &name) {
    int32_t idx = index(name);
    if (idx < 0) {
        return -1;
    }
    return numel(idx);
}

int EngineImplement::numel(int ibinding) {
    if (ibinding < 0 || ibinding >= input_node_dims_.size()+output_node_dims_.size()) {
        return -1;
    }
    if (ibinding < input_node_dims_.size()) {
        return numel(input_node_dims_[ibinding]);
    } else {
        return numel(output_node_dims_[ibinding-input_node_dims_.size()]);
    }
}

int EngineImplement::num_bindings() {
    return input_node_names_.size() + output_node_names_.size();
}

bool EngineImplement::is_input(int ibinding) {
    return ibinding < input_node_names_.size();
}

bool EngineImplement::set_run_dims(const std::string &name, const std::vector<int> &dims) {
    int32_t idx = index(name);
    if (idx < 0) {
        return false;
    }
    return set_run_dims(idx, dims);
}

bool EngineImplement::set_run_dims(int ibinding, const std::vector<int> &dims) {
    throw std::runtime_error("Not implemented");
}

DType EngineImplement::dtype(const std::string &name) {
    int32_t idx = index(name);
    if (idx < 0) {
        throw std::runtime_error("Invalid tensor name");
    }
    return dtype(idx);
}

DType type(ONNXTensorElementDataType type){
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return DType::FLOAT;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return DType::HALF;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return DType::INT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return DType::INT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return DType::UINT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            throw std::runtime_error("Unsupported data type");
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return DType::BOOL;
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

DType EngineImplement::dtype(int ibinding) {
    if (ibinding < 0 || ibinding >= input_node_dims_.size()+output_node_dims_.size()) {
        throw std::runtime_error("Invalid tensor index");
    }
    if (ibinding < input_node_dims_.size()) {
        return  type(session_.GetInputTypeInfo(ibinding).GetTensorTypeAndShapeInfo().GetElementType());
    } else {
        return type(session_.GetOutputTypeInfo(ibinding-input_node_dims_.size()).GetTensorTypeAndShapeInfo().GetElementType());
    }
}

bool EngineImplement::has_dynamic_dim() {
    for (auto& dim : input_node_dims_) {
        for (auto& d : dim) {
            if (d < 0) {
                return true;
            }
        }
    }
    for (auto& dim : output_node_dims_) {
        for (auto& d : dim) {
            if (d < 0) {
                return true;
            }
        }
    }
    return false;
}

void EngineImplement::print(const char *name) {
    std::cout << "Engine: " << name << std::endl;
    std::cout << "  Inputs:" << std::endl;
    for (size_t i = 0; i < input_node_names_.size(); ++i) {
        std::cout << "    " << input_node_names_[i] << " : " << format_shape(input_node_dims_[i]) << std::endl;
    }
    std::cout << "  Outputs:" << std::endl;
    for (size_t i = 0; i < output_node_names_.size(); ++i) {
        std::cout << "    " << output_node_names_[i] << " : " << format_shape(output_node_dims_[i]) << std::endl;
    }
}


EngineImplement::~EngineImplement() {
    // 释放输入和输出节点名称
    for (auto name : input_node_names_) {
        allocator_.Free(const_cast<char*>(name));
    }
    for (auto name : output_node_names_) {
        allocator_.Free(const_cast<char*>(name));
    }
}

void EngineImplement::initializeIO() {
    size_t num_input_nodes = session_.GetInputCount();
    input_node_names_.resize(num_input_nodes);
    input_node_dims_.resize(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        Ort::AllocatedStringPtr input_name = session_.GetInputNameAllocated(i, allocator_);
        size_t input_name_len = std::strlen(input_name.get());
        char* alloc_input_name = reinterpret_cast<char*>(allocator_.Alloc(input_name_len + 1));
        std::copy(input_name.get(), input_name.get() + input_name_len, alloc_input_name);
        alloc_input_name[input_name_len] = '\0';
        input_node_names_[i] = alloc_input_name;

        Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_dims_[i] = tensor_info.GetShape();
    }

    size_t num_output_nodes = session_.GetOutputCount();
    output_node_names_.resize(num_output_nodes);
    output_node_dims_.resize(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; i++) {
        Ort::AllocatedStringPtr output_name = session_.GetOutputNameAllocated(i, allocator_);
        size_t output_name_len = std::strlen(output_name.get());
        char* alloc_output_name = reinterpret_cast<char*>(allocator_.Alloc(output_name_len + 1));
        std::copy(output_name.get(), output_name.get() + output_name_len, alloc_output_name);
        alloc_output_name[output_name_len] = '\0';
        output_node_names_[i] = alloc_output_name;

        Ort::TypeInfo type_info = session_.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        tensor_info.GetElementType();
        output_node_dims_[i] = tensor_info.GetShape();
    }
}

bool EngineImplement::forward(const std::vector<const void*>& bindings, void* stream, void* input_consum_event) {
   // 转换 bindings 为 Ort::Value 张量
    std::vector<Ort::Value> input_tensors;
    for (size_t i = 0; i < bindings.size(); ++i) {
        const float* input_data = static_cast<const float*>(bindings[i]);
        const std::vector<int64_t>& dims = input_node_dims_[i];

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            allocator_info_, const_cast<float*>(input_data),
            numel(dims), dims.data(), dims.size());

        input_tensors.emplace_back(std::move(input_tensor));
    }

    // 运行推理
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_node_names_.data(),
        output_node_names_.size());

    // 处理 output_tensors，根据需要将输出数据复制到 bindings 中

    return true;
}

std::shared_ptr<Engine> load(const std::string &file) {
  std::shared_ptr<EngineImplement> impl = std::make_shared<EngineImplement>(file);

  return impl;
}

// 实现其他方法...

};  // namespace TensorRT
