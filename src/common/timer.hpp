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

#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include "check.hpp"
//#include <iostream>
//#include <mv_stream/slog.hpp>

//static auto LOG = mv::Log::getLogger("Timer");

namespace fastbev {

class EventTimer {
 public:
  EventTimer(): start_(std::chrono::high_resolution_clock::now()) {

  }

  virtual ~EventTimer() {

  }

  void start() { start_ = std::chrono::high_resolution_clock::now(); }

  float stop(const char* prefix = "timer") {
    float ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - start_).count();

    //std::cout<< "[⏰ " << prefix << "]: \t" << ms << " ms";

    return ms;
  }

 private:
  std::chrono::high_resolution_clock::time_point start_;
};

};  // namespace nv

#endif  // __TIMER_HPP__