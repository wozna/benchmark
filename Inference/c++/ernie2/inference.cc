// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <numeric>
#include <paddle_inference_api.h>
#include <sstream>
#include <string>
#include <vector>

#ifdef ENABLE_PADDLE_PROFILER
#include <paddle/fluid/platform/profiler.h>
DECLARE_bool(profile);
#endif

DEFINE_string(model_dir, "", "model directory");
DEFINE_string(data, "", "input data path");
DEFINE_int32(repeat, 1, "repeat");
DEFINE_bool(output_prediction, false,
            "Whether to output the prediction results.");
DEFINE_bool(use_gpu, false, "Whether to use GPU for prediction.");
DEFINE_bool(use_int8, false, "predict int8 model");
DEFINE_bool(squash, false, "squash fc+dequantize");
DEFINE_bool(remove_scale, false, "remove scale operator");
DEFINE_bool(enable_memory_optim, false, "enable memory optimization");
DEFINE_bool(short, false, "run only a few iterations");
DEFINE_bool(debug, false, "generate .dot file after each pass");
DEFINE_int32(device, 0, "device.");
DEFINE_int32(num_threads, 1, "num_threads");

template <typename T> void GetValueFromStream(std::stringstream *ss, T *t) {
  (*ss) >> (*t);
}

template <>
void GetValueFromStream<std::string>(std::stringstream *ss, std::string *t) {
  *t = ss->str();
}

// Split string to vector
template <typename T>
void Split(const std::string &line, char sep, std::vector<T> *v) {
  std::stringstream ss;
  T t;
  for (auto c : line) {
    if (c != sep) {
      ss << c;
    } else {
      GetValueFromStream<T>(&ss, &t);
      v->push_back(std::move(t));
      ss.str({});
      ss.clear();
    }
  }

  if (!ss.str().empty()) {
    GetValueFromStream<T>(&ss, &t);
    v->push_back(std::move(t));
    ss.str({});
    ss.clear();
  }
}

template <typename T> constexpr paddle::PaddleDType GetPaddleDType();

template <> constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
  return paddle::PaddleDType::INT64;
}

template <> constexpr paddle::PaddleDType GetPaddleDType<float>() {
  return paddle::PaddleDType::FLOAT32;
}

// Parse tensor from string
template <typename T>
bool ParseTensor(const std::string &field, paddle::PaddleTensor *tensor) {
  std::vector<std::string> data;
  Split(field, ':', &data);
  if (data.size() < 2)
    return false;

  std::string shape_str = data[0];

  std::vector<int> shape;
  Split(shape_str, ' ', &shape);

  std::string mat_str = data[1];

  std::vector<T> mat;
  Split(mat_str, ' ', &mat);

  tensor->shape = shape;
  auto size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      sizeof(T);
  tensor->data.Resize(size);
  std::copy(mat.begin(), mat.end(), static_cast<T *>(tensor->data.data()));
  tensor->dtype = GetPaddleDType<T>();

  return true;
}

// Parse input tensors from string
bool ParseLine(const std::string &line,
               std::vector<paddle::PaddleTensor> *tensors) {
  std::vector<std::string> fields;
  Split(line, ';', &fields);

  // if (fields.size() != 7) return false;

  tensors->clear();
  tensors->reserve(4);

  int i = 0;
  // src_ids
  paddle::PaddleTensor src_ids;
  ParseTensor<int64_t>(fields[i++], &src_ids);
  src_ids.name = "eval_placeholder_0";
  tensors->push_back(src_ids);

  // pos_ids
  paddle::PaddleTensor pos_ids;
  ParseTensor<int64_t>(fields[i++], &pos_ids);
  pos_ids.name = "eval_placeholder_1";
  tensors->push_back(pos_ids);

  /*
    // src_ids
    paddle::PaddleTensor src_ids;
    ParseTensor<int64_t>(fields[i++], &src_ids);
    src_ids.name = "eval_placeholder_0";
    tensors->push_back(src_ids);

    // pos_ids
    paddle::PaddleTensor pos_ids;
    ParseTensor<int64_t>(fields[i++], &pos_ids);
    pos_ids.name = "eval_placeholder_1";
    tensors->push_back(pos_ids);

    // sent_ids
    paddle::PaddleTensor sent_ids;
    ParseTensor<int64_t>(fields[i++], &sent_ids);
    sent_ids.name = "eval_placeholder_2";
    tensors->push_back(sent_ids);

    // input_mask
    paddle::PaddleTensor input_mask;
    ParseTensor<float>(fields[i++], &input_mask);
    input_mask.name = "eval_placeholder_3";
    tensors->push_back(input_mask);
  */

  return true;
}

// Print outputs to log
void PrintOutputs(const std::vector<paddle::PaddleTensor> &outputs) {
  for (size_t i = 0; i < outputs.front().data.length() / sizeof(float) / 3;
       i += 1) {
    std::cout << static_cast<float *>(outputs[0].data.data())[3 * i] << "\t"
              << static_cast<float *>(outputs[0].data.data())[3 * i + 1] << "\t"
              << static_cast<float *>(outputs[0].data.data())[3 * i + 2]
              << std::endl;
  }
}

// void PrintOutputs(const std::vector<paddle::PaddleTensor> &outputs) {
//  for (size_t i = 0; i < outputs.front().data.length() / sizeof(float) / 2; i
//  += 1) {
//    std::cout << static_cast<float *>(outputs[0].data.data())[2 * i] << "\t"
//         << static_cast<float *>(outputs[0].data.data())[2 * i + 1] <<
//         std::endl;
//  }
//}

bool LoadInputData(std::vector<std::vector<paddle::PaddleTensor>> *inputs) {
  if (FLAGS_data.empty()) {
    LOG(ERROR) << "please set input data path";
    return false;
  }

  std::ifstream fin(FLAGS_data);
  std::string line;

  int lineno = 0;
  while (std::getline(fin, line)) {
    std::vector<paddle::PaddleTensor> feed_data;
    if (!ParseLine(line, &feed_data)) {
      LOG(ERROR) << "Parse line[" << lineno << "] error!";
    } else {
      inputs->push_back(std::move(feed_data));
    }
  }

  return true;
}

// Bert inference demo
// Options:
//     --model_dir: bert model file directory
//     --data: data path
//     --repeat: repeat num
//     --use_gpu: use gpu
int main(int argc, char *argv[]) {
  google::InitGoogleLogging(*argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_dir.empty()) {
    LOG(ERROR) << "please set model dir";
    return -1;
  }

  paddle::AnalysisConfig config;
  config.SetModel(FLAGS_model_dir);
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
    config.EnableCUDNN();
  } else {
    config.DisableGpu();
    config.EnableMKLDNN();
    config.SetCpuMathLibraryNumThreads(FLAGS_num_threads);
  }
  config.SwitchSpecifyInputNames(false);
  if (FLAGS_use_int8) {
    std::cout << "--- use int8 ---" << std::endl;
    config.pass_builder()->AppendPass("fc_mkldnn_pass");
  } else {
    std::cout << "--- use fp32 ---" << std::endl;
  }
  if (!FLAGS_remove_scale) {
    std::cout << "--- keep scale ---" << std::endl;
    config.pass_builder()->DeletePass("reshape_transpose_scale_pass");
    config.pass_builder()->DeletePass("dequant_scale_pass");
  } else {
    std::cout << "--- remove scale ---" << std::endl;
  }
  if (FLAGS_enable_memory_optim) {
    std::cout << "--- enable memory optim ---" << std::endl;
    config.EnableMemoryOptim();
  } else {
    std::cout << "--- disabled memory optim ---" << std::endl;
  }
  if (FLAGS_squash) {
    std::cout << "--- squash fc+dequantize ---" << std::endl;
    config.pass_builder()->AppendPass("cpu_quantize_squash_pass");
  }

  // config->pass_builder()->DeletePass("fc_fuse_pass");
  config.SwitchIrOptim(true);
  if (FLAGS_debug)
    config.SwitchIrDebug(true);

  auto predictor = CreatePaddlePredictor(config);

  std::vector<std::vector<paddle::PaddleTensor>> inputs;
  if (!LoadInputData(&inputs)) {
    LOG(ERROR) << "load input data error!";
    return -1;
  }

#ifdef ENABLE_PADDLE_PROFILER
  if (FLAGS_profile) {
    if (!FLAGS_use_gpu) {
      std::cout << "---profiler enabled---" << std::endl;
      paddle::platform::EnableProfiler(paddle::platform::ProfilerState::kCPU);
    } else {
      paddle::platform::EnableProfiler(paddle::platform::ProfilerState::kAll);
    }
  }
  if (FLAGS_profile) {
    paddle::platform::ResetProfiler();
  }
#endif

  std::vector<paddle::PaddleTensor> fetch;
  int total_time{0};
  // auto predict_timer = []()
  int num_samples{0};
  int count{0};
  for (int i = 0; i < 100; i++) { // warm up
    predictor->Run(inputs[0], &fetch);
  }
  for (int i = 0; i < FLAGS_repeat; i++) {
    for (int id = 0; id < inputs.size(); ++id) {
      if (FLAGS_short && id == 2)
        break;
      fetch.clear();
      std::cout << "--- iteration " << id << " ---" << std::endl;
      auto start = std::chrono::system_clock::now();
      predictor->Run(inputs[id], &fetch);
      auto end = std::chrono::system_clock::now();
      if (FLAGS_output_prediction && i == 0) {
        PrintOutputs(fetch);
      }
      count += 1;
      if (!fetch.empty()) {
        total_time +=
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        // num_samples += fetch.front().data.length() / 2 / sizeof(float);
        // num_samples += fetch.front().data.length() / (sizeof(float) * 3);
        num_samples++;
      }
    }
  }

#ifdef ENABLE_PADDLE_PROFILER
  if (FLAGS_profile) {
    paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                      "ernie.inference.profile");
  }
#endif

  auto per_sample_ms = static_cast<float>(total_time) / num_samples;
  fprintf(stderr, "averay latency: %f timecost %f\n", per_sample_ms,
          static_cast<float>(total_time));
  LOG(INFO) << "Run " << num_samples
            << " samples, average latency: " << per_sample_ms
            << "ms per sample.";
  LOG(INFO) << count;

  return 0;
}
