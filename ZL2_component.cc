/******************************************************************************
 * Copyright 2023 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/******************************************************************************
 * @file ZL2_component.cc
 *****************************************************************************/

#include "ZL2/ZL2_component.h"

namespace apollo {

// 定义 ZL2 类的 Init() 初始化函数
// 作用：组件启动时 自动执行一次（只跑一次）
bool ZL2::Init() {

  // 从配置文件加载 protobuf 配置到 config_
  // ACHECK = Apollo断言，加载失败直接报错退出
  ACHECK(ComponentBase::GetProtoConfig(&config_))
      << "failed to load ZL2 config file "
      << ComponentBase::ConfigFilePath();

  // 打印日志：加载配置成功
  AINFO << "Load config succedded.\n" << config_.DebugString();

  // 打印日志：初始化成功
  AINFO << "Init ZL2 succedded.";

  // 返回true：告诉CyberRT“初始化成功”
  return true;
}

// 定义 Proc() 消息处理函数
// 作用：**只要通道收到消息，就自动调用这个函数！**
// msg0：收到的消息（类型是 ZL2Msg，就是你proto里的空消息）
bool ZL2::Proc(const std::shared_ptr<apollo::ZL2Msg>& msg0) {

  // 打印日志：收到消息
  AINFO << "message recieved.\n" << msg0->DebugString();

  // 返回true：处理成功
  return true;
}

} // namespace apollo