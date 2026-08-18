#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include <libserial/SerialStream.h>
#include <string>
#include <sstream>
#include <vector>
#include <cctype>

class SerialPortPublisher : public rclcpp::Node {
public:
    SerialPortPublisher() : Node("xycar_ultrasonic") {
        publisher_ = this->create_publisher<std_msgs::msg::Int32MultiArray>("xycar_ultrasonic", 10);
        serial_.Open("/dev/ttySonic");
        serial_.SetBaudRate(LibSerial::BaudRate::BAUD_115200);
        serial_.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8);
        serial_.SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE);
        serial_.SetParity(LibSerial::Parity::PARITY_NONE);
        serial_.SetStopBits(LibSerial::StopBits::STOP_BITS_1);

        if (!serial_.IsOpen()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open the serial port.");
            rclcpp::shutdown();
        }
        // Initialize the message
        msg_.data.resize(8, 0);

        // Start the data publishing loop
        publishData();
    }

    void publishData() {
        rclcpp::Rate loop_rate(30);  // Create a rate object for publishing as fast as possible

        while (rclcpp::ok()) {
            std::string serialData = readSerialData();
            if (!serialData.empty()) {
                msg_.data = parseStringData(serialData);
                publisher_->publish(msg_);
            }
            loop_rate.sleep();  // Sleep to yield the rest of the time
        }
    }

    std::string readSerialData() {
        std::string data;
        while (serial_.IsDataAvailable()) {
            char c;
            serial_.read(&c, 1);
            data += c;
        }
        return data;
    }

    std::vector<int32_t> parseStringData(const std::string& data) {
        std::vector<int32_t> values;
        std::istringstream ss(data);
        std::string token;

        for (int i = 0; i < 8; i++) {
            if (std::getline(ss, token, ',')) {
                int32_t value = 0; // Initialize to 0

                if (!token.empty() && isdigit(token[0])) {
                    value = std::stoi(token);
                    if (value < 0 || value > 140) {
                        value = 0;
                    }
                }
                values.push_back(value);
            }
        }

        return values;
    }

private:
    rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr publisher_;
    std_msgs::msg::Int32MultiArray msg_;
    LibSerial::SerialStream serial_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SerialPortPublisher>());
    rclcpp::shutdown();
    return 0;
}

