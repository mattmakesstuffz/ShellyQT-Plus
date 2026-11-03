#include "engine.hpp"

#include <iostream>
#include <string>

int main() {
    std::cout << "============================\n";
    std::cout << "       ShellyQT+\n";
    std::cout << "    25M C++ Neural Chatbot\n";
    std::cout << "============================\n\n";

    ShellyEngine shelly;

    if (!shelly.load(
        "tokenizer/shelly.model",
        "weights/shelly_25m.bin"
    )) {
        std::cerr << "Failed to load ShellyQT+!\n";
        return 1;
    }

    std::cout << "\nShellyQT+ is ready! Type /exit to quit.\n\n";

    std::string input;

    while (true) {
        std::cout << "You: ";

        if (!std::getline(std::cin, input))
            break;

        if (input == "/exit")
            break;

        if (input.empty())
            continue;

        try {
            std::string prompt = "User: " + input + "\nShelly:";

            std::cout << "Shelly: "
                      << shelly.generate(prompt, 60)
                      << "\n\n";
        }
        catch (const std::exception& error) {
            std::cerr << "Generation error: "
                      << error.what() << "\n";
        }
    }

    std::cout << "Shelly: Bye! :)\n";
    return 0;
}
