#include <cyber/cyber.h>
#include "ZL2/proto/ZL2.pb.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <thread>

// 配置宏 - 可自定义修改这些值
#define SERIAL_PORT           "/dev/ttyUSB3"  // 串口设备
#define BAUD_RATE            B115200         // 波特率
#define DISTANCE_THRESHOLD_MM 400.0f        // 距离阈值(mm)
#define CENTER_ANGLE         180.0f          // 中心角度(正前方)
#define LEFT_RANGE_DEG        30.0f          // 左侧检测范围(度)
#define RIGHT_RANGE_DEG       30.0f          // 右侧检测范围(度)            

using apollo::ZL2::proto::Perception;

void delay(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

bool kbhit() {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return true;
    }
    return false;
}

// 雷达数据包结构
struct LidarDataPacket {
    uint16_t header;        // 数据包头 0x55AA
    uint8_t packetType;     // 包类型
    uint8_t sampleCount;    // 采样数量
    uint16_t startAngle;    // 起始角
    uint16_t endAngle;      // 结束角
    uint16_t checksum;      // 校验码
    std::vector<uint16_t> samples; // 采样数据
};

// 解析后的点云数据
struct PointData {
    float angle;    // 角度(度)
    float distance; // 距离(mm)
    bool isLeft;    // 是否在左侧
};

// 串口通信类
class SerialPort {
private:
    int fd; // 串口文件描述符

public:
    SerialPort() : fd(-1) {}

    ~SerialPort() {
        if (fd != -1) close(fd);
    }

    bool openPort(const char* port, speed_t baud) {
        fd = open(port, O_RDWR | O_NOCTTY);
        if (fd == -1) {
            perror("openPort: Unable to open port");
            return false;
        }

        // 配置串口参数
        struct termios options;
        tcgetattr(fd, &options);
        cfsetispeed(&options, baud);
        cfsetospeed(&options, baud);

        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_oflag &= ~OPOST;

        options.c_cc[VMIN] = 0;
        options.c_cc[VTIME] = 10;

        if (tcsetattr(fd, TCSANOW, &options) != 0) {
            perror("openPort: Failed to set port attributes");
            close(fd);
            fd = -1;
            return false;
        }

        return true;
    }

    int readData(uint8_t* buffer, size_t size) {
        if (fd == -1) return -1;
        return read(fd, buffer, size);
    }
};

// 雷达数据处理类
class LidarProcessor {
    private:
        int dir_data = 0;
        int DIR_M_TAB = 0;
        float distanceThreshold_;
        float centerAngle_;
        float leftRange_;
        float rightRange_;
        std::vector<PointData> pointCloud_;
        std::shared_ptr<apollo::cyber::Writer<Perception>> writer_;
    
    public:
        LidarProcessor(std::shared_ptr<apollo::cyber::Writer<Perception>> writer) 
            : writer_(writer){
            // 使用宏定义的默认值初始化
            distanceThreshold_ = DISTANCE_THRESHOLD_MM;
            centerAngle_ = CENTER_ANGLE;
            leftRange_ = LEFT_RANGE_DEG;
            rightRange_ = RIGHT_RANGE_DEG;
        }
    
        // 处理接收到的数据包
        bool processPacket(const uint8_t* data, size_t length) {
            if (length < 10) return false; // 最小包长度检查
    
            LidarDataPacket packet;
            
            // 解析包头
            packet.header = (data[1] << 8) | data[0];
            if (packet.header != 0x55AA) {
                std::cerr << "Invalid packet header: " << std::hex << packet.header << std::endl;
                return false;
            }
    
            // 解析包类型
            packet.packetType = data[2];
            
            // 解析采样数量
            packet.sampleCount = data[3];
            
            // 解析起始角和结束角
            packet.startAngle = (data[5] << 8) | data[4];
            packet.endAngle = (data[7] << 8) | data[6];
            
            // 解析校验码
            packet.checksum = (data[9] << 8) | data[8];
            
            // 检查数据长度是否足够
            size_t expectedLength = 10 + packet.sampleCount * 2;
            if (length < expectedLength) {
                std::cerr << "Incomplete packet: expected " << expectedLength 
                          << " bytes, got " << length << std::endl;
                return false;
            }
            
            // 解析采样数据
            for (int i = 0; i < packet.sampleCount; ++i) {
                int offset = 10 + i * 2;
                uint16_t sample = (data[offset + 1] << 8) | data[offset];
                packet.samples.push_back(sample);
            }
    
            // 处理数据包
            return processLidarPacket(packet);
        }
    
        // 设置距离阈值
        void setDistanceThreshold(float threshold) {
            distanceThreshold_ = threshold;
        }
    
        // 设置中心角度
        void setCenterAngle(float angle) {
            centerAngle_ = angle;
        }
    
        // 设置左右检测范围
        void setDetectionRanges(float leftRange, float rightRange) {
            leftRange_ = leftRange;
            rightRange_ = rightRange;
        }
    
    private:
        // 处理雷达数据包
        bool processLidarPacket(const LidarDataPacket& packet) {
            // 检查是否是起始数据包
            bool isStartPacket = (packet.packetType & 0x01) == 1;
            
            if (isStartPacket) {
                // 如果是起始包，可以获取扫描频率等信息
                return true;
            }
    
            // 计算角度数据
            float startAngle = (packet.startAngle >> 1) / 64.0f;
            float endAngle = (packet.endAngle >> 1) / 64.0f;
            
            // 计算角度差
            float angleDiff = endAngle - startAngle;
            if (angleDiff < 0) angleDiff += 360.0f;
            
            // 计算每个采样点的角度
            for (int i = 0; i < packet.sampleCount; ++i) {
                PointData point;
                
                // 计算角度(一级解析)
                if (packet.sampleCount == 1) {
                    point.angle = startAngle;
                } else {
                    point.angle = startAngle + (angleDiff / (packet.sampleCount - 1)) * i;
                }
                
                // 归一化角度到0-360度范围
                point.angle = fmod(point.angle, 360.0f);
                if (point.angle < 0) point.angle += 360.0f;
                
                // 计算距离
                point.distance = packet.samples[i] / 4.0f;
                
                // 添加角度修正(二级解析)
                if (point.distance > 0) {
                    float angleCorrection = atan(21.8f * (155.3f - point.distance) / (155.3f * point.distance));
                    angleCorrection = angleCorrection * 180.0f / M_PI;
                    point.angle += angleCorrection;
                    
                    // 再次归一化角度
                    point.angle = fmod(point.angle, 360.0f);
                    if (point.angle < 0) point.angle += 360.0f;
                }
                
                // 判断点在左侧还是右侧
                point.isLeft = isPointInLeftRange(point.angle);
                
                pointCloud_.push_back(point);
            }
            
            // 检查点云数据是否完整(假设一圈扫描完成)
            if (pointCloud_.size() > 100) {
                detectLeftRightObstacles();
                pointCloud_.clear();
                return true;
            }
            
            return false;
        }
    
        // 判断点是否在左侧检测范围内
        bool isPointInLeftRange(float angle) {
            float leftStart = centerAngle_ - leftRange_;
            float leftEnd = centerAngle_;
            
            // 处理角度范围跨越0度的情况
            if (leftStart < 0) leftStart += 360.0f;
            
            if (leftStart < leftEnd) {
                return (angle >= leftStart && angle < leftEnd);
            } else {
                return (angle >= leftStart || angle < leftEnd);
            }
        }
    
        // 判断点是否在右侧检测范围内
        bool isPointInRightRange(float angle) {
            float rightStart = centerAngle_;
            float rightEnd = centerAngle_ + rightRange_;
            
            // 处理角度范围跨越360度的情况
            if (rightEnd > 360.0f) rightEnd -= 360.0f;
            
            if (rightStart < rightEnd) {
                return (angle >= rightStart && angle < rightEnd);
            } else {
                return (angle >= rightStart || angle < rightEnd);
            }
        }
    
        // 检测左右两侧障碍物
        void detectLeftRightObstacles() {
            struct ObstacleInfo {
                bool detected;
                int pointCount;
                float minDistance;
                float avgDistance;
                std::vector<float> distances;
                
                ObstacleInfo() : detected(false), pointCount(0), 
                                minDistance(std::numeric_limits<float>::max()),
                                avgDistance(0.0f) {}
            };
            
            ObstacleInfo left, right;
            
            for (const auto& point : pointCloud_) {
                if (point.distance <= 0 || point.distance > distanceThreshold_) {
                    continue;
                }
                
                if (isPointInLeftRange(point.angle)) {
                    left.detected = true;
                    left.pointCount++;
                    left.distances.push_back(point.distance);
                    if (point.distance < left.minDistance) {
                        left.minDistance = point.distance;
                    }
                } else if (isPointInRightRange(point.angle)) {
                    right.detected = true;
                    right.pointCount++;
                    right.distances.push_back(point.distance);
                    if (point.distance < right.minDistance) {
                        right.minDistance = point.distance;
                    }
                }
            }
            
            // 计算平均距离
            if (left.pointCount > 0) {
                float sum = 0.0f;
                for (auto d : left.distances) sum += d;
                left.avgDistance = sum / left.distances.size();
            }
            
            if (right.pointCount > 0) {
                float sum = 0.0f;
                for (auto d : right.distances) sum += d;
                right.avgDistance = sum / right.distances.size();
            }
            
            // 输出检测结果
            std::cout << "\nDetection Results (Center: " << centerAngle_ << "°):" << std::endl;
            
            // 左侧检测范围显示
            float leftStart = centerAngle_ - leftRange_;
            if (leftStart < 0) leftStart += 360.0f;
            std::cout << "Left Side (" << leftStart << "° to " << centerAngle_ << "°):" << std::endl;
            std::cout << "  Obstacle detected: " << (left.detected ? "Yes" : "No") << std::endl;
            if (left.detected) {
                std::cout << "  Points: " << left.pointCount << std::endl;
                std::cout << "  Min distance: " << left.minDistance << "mm" << std::endl;
                std::cout << "  Avg distance: " << left.avgDistance << "mm" << std::endl;
            }
            
            // 右侧检测范围显示
            float rightEnd = centerAngle_ + rightRange_;
            if (rightEnd > 360.0f) rightEnd -= 360.0f;
            std::cout << "Right Side (" << centerAngle_ << "° to " << rightEnd << "°):" << std::endl;
            std::cout << "  Obstacle detected: " << (right.detected ? "Yes" : "No") << std::endl;
            if (right.detected) {
                std::cout << "  Points: " << right.pointCount << std::endl;
                std::cout << "  Min distance: " << right.minDistance << "mm" << std::endl;
                std::cout << "  Avg distance: " << right.avgDistance << "mm" << std::endl;
            }
            
            std::cout << "-----------------" << std::endl;
            auto msg = std::make_shared<Perception>();
            if((left.pointCount>right.pointCount)&&(left.pointCount>4)){
                dir_data = 2;
            }else if((left.pointCount<right.pointCount)&&(right.pointCount>10)){
                dir_data = 1;
            }else{
                DIR_M_TAB++;
                if(DIR_M_TAB>20){
                    dir_data = 0;
                    DIR_M_TAB = 0;
                }
            }
            msg->set_radar(dir_data);
            writer_->Write(msg);
            std::cout << "  Direction: " << dir_data << std::endl;
            
            // 创建并发送消息
            
            
        }
    };
    
    int main(int argc, char *argv[]) {
        // 初始化cyber
        apollo::cyber::Init("radar");
        // 创建node
        auto talker_node = apollo::cyber::CreateNode("radar_writer_1");
        // 创建writer，写Chatter类型消息
        auto talker = talker_node->CreateWriter<Perception>("/apollo/radar/data");

        // 创建并打开串口
        const char* serial_path = argv[1];
        std::cerr << "串口序号: " << argv[1] << std::endl;
        SerialPort serial;
        if (!serial.openPort(serial_path, BAUD_RATE)) {
            std::cerr << "Failed to open serial port " << SERIAL_PORT << std::endl;
            return 1;
        }
        std::cout << "Successfully opened serial port " << SERIAL_PORT << std::endl;
    
        // 创建雷达处理器(使用宏定义的默认值)
        LidarProcessor processor(talker);
        
        std::cout << "Initial Configuration:" << std::endl;
        std::cout << "Distance Threshold: " << DISTANCE_THRESHOLD_MM << "mm" << std::endl;
        std::cout << "Center Angle: " << CENTER_ANGLE << "° (Front)" << std::endl;
        std::cout << "Left Detection Range: " << LEFT_RANGE_DEG << "° (" 
                  << (CENTER_ANGLE - LEFT_RANGE_DEG) << "° to " << CENTER_ANGLE << "°)" << std::endl;
        std::cout << "Right Detection Range: " << RIGHT_RANGE_DEG << "° (" 
                  << CENTER_ANGLE << "° to " << (CENTER_ANGLE + RIGHT_RANGE_DEG) << "°)" << std::endl;
        std::cout << "-----------------" << std::endl;
        std::cout << "Starting to read data from lidar..." << std::endl;
    
        // 数据接收缓冲区
        uint8_t buffer[256];
        std::vector<uint8_t> packetBuffer;
        bool inPacket = false;
    
        while (apollo::cyber::OK()) {
            if (kbhit()) {
                char key = getchar();
                if (toupper(key) == 'Q') {
                    std::cout << "退出程序..." << std::endl;
                    break;
                }
            }   

            // 从串口读取数据
            int bytesRead = serial.readData(buffer, sizeof(buffer));
            if (bytesRead <= 0) {
                if (bytesRead == -1) {
                    perror("Error reading from serial port");
                    break;
                }
                continue;
            }
    
            // 处理接收到的数据
            for (int i = 0; i < bytesRead; i++) {
                if (!inPacket) {
                    // 寻找包头 (0xAA 0x55)
                    if (i + 1 < bytesRead && buffer[i] == 0xAA && buffer[i+1] == 0x55) {
                        inPacket = true;
                        packetBuffer.clear();
                        packetBuffer.push_back(buffer[i]);
                        packetBuffer.push_back(buffer[i+1]);
                        i++; // 跳过第二个字节
                        continue;
                    }
                } else {
                    // 收集数据包
                    packetBuffer.push_back(buffer[i]);
                    
                    // 检查是否收集到完整的数据包
                    if (packetBuffer.size() >= 10) {
                        uint8_t sampleCount = packetBuffer[3];
                        size_t expectedLength = 10 + sampleCount * 2;
                        
                        if (packetBuffer.size() >= expectedLength) {
                            // 处理完整的数据包
                            processor.processPacket(packetBuffer.data(), packetBuffer.size());
                            inPacket = false;
                            
                            // 如果有剩余数据，保留在缓冲区中
                            if (packetBuffer.size() > expectedLength) {
                                std::vector<uint8_t> remaining(
                                    packetBuffer.begin() + expectedLength, 
                                    packetBuffer.end()
                                );
                                packetBuffer = remaining;
                                inPacket = true; // 继续处理剩余数据
                            } else {
                                packetBuffer.clear();
                            }
                        }
                    }
                }
            }
        }
    
        return 0;
    }