#include "common/http.h"
#include "llama.h"
#include "vendor/nlohmann/json.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <io.h>
#    define ISATTY _isatty
#    define FILENO _fileno
#else
#    include <unistd.h>
#    define ISATTY isatty
#    define FILENO fileno
#endif

using json = nlohmann::ordered_json;

const std::string sp = R"(You are llama-copilot, a concise CLI assistant.
You receive structured shell context.
When a target is specified, operate ONLY on that target.
Treat history and environment as background evidence.)";

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s [-s/--stream] [-n n_predict] [-f context_file] -u http://localhost:8033 prompt\n", argv[0]);
    printf("\n");
}

static std::string read_from_stdin(bool & success) {
    success = false;
    std::string stdin_context;

    // read context from stdin
    std::stringstream stdin_buffer;

    stdin_buffer << std::cin.rdbuf();
    if (std::cin.fail()) {
        fprintf(stderr, "Error: could not read the entire standard input.\n");
        return std::string();
    }

    std::string stdin_buffer_str = stdin_buffer.str();
    if (stdin_buffer_str.length() < 400) {
        stdin_context ="<STDIN mime='text/plain' source='pipe'>\n" + stdin_buffer_str + "\n</STDIN>";  // first n_ctx/4

    } else {
        stdin_context = "<STDIN mime='text/plain' source='pipe' truncated='true' header='true'>\n" + stdin_buffer_str.substr(0, 100) + "\n</STDIN>\n";  // first n_ctx/4
        stdin_context += "<STDIN mime='text/plain' source='pipe' truncated='true' tail='true'>\n" +
                         stdin_buffer_str.substr(stdin_buffer_str.length() - 300, 300) +
                         "\n</STDIN>\n";  // last n_ctx/4
    }
    success = true;
    return stdin_context;
}

static std::string get_cwd() {
    char buffer[1024];
#ifdef _WIN32
    GetModuleFileNameA(NULL, buffer, sizeof(buffer));
#else
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
    } else {
        buffer[0] = '\0';
    }
#endif
    return std::string(buffer);
}

static std::string get_os() {
#ifdef _WIN32
    return "Windows";
#elif __APPLE__
    return "macOS";
#elif __linux__
    return "Linux";
#else
    return "Unknown OS";
#endif
}

static std::string get_shell_name() {
    const char * shell = std::getenv("SHELL");
    if (shell) {
        std::string shell_str(shell);
        return std::filesystem::path(shell_str).filename().string();
    }
    return "unknown";
}

static std::string read_from_file(const char * filepath, bool & success, int limit_bytes = -1) {
    success = false;

    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        fprintf(stderr, "%s: could not open file '%s' for reading: %s\n", __func__, filepath, strerror(errno));
        return std::string();
    }
    // do not assume the file is seekable (e.g. /dev/stdin)
    std::stringstream buffer;
    buffer << in.rdbuf();
    if (in.fail()) {
        fprintf(stderr, "%s: could not read the entire file '%s': %s\n", __func__, filepath, strerror(errno));
        return std::string();
    }

    success = true;
    if (limit_bytes < 0) {
        return buffer.str();
    }

    return buffer.str().substr(0, limit_bytes);
}

static std::string get_terminal_history_N_lines(int how_many_lines) {
    if (how_many_lines <= 0) {
        return "";
    }

    std::filesystem::path history_file;
    if (get_shell_name() == "bash") {
        // resolve to default bash history file
        history_file = std::filesystem::path(getenv("HOME")) / ".bash_history";
    } else if (get_shell_name() == "zsh") {
        history_file = std::filesystem::path(getenv("HOME")) / ".zsh_history";
    } else {
        fprintf(stderr, "Unsupported shell for history retrieval: %s\n", get_shell_name().c_str());
        return "";
    }

    bool success;
    std::string content = read_from_file(history_file.c_str(), success);
    if (!success) {
        fprintf(stderr, "FERROR: read history failed\n");
        return "";
    }

    std::vector<std::string> history_N_lines = {};
    size_t                   pos             = 0;
    for (int i = 0; i < how_many_lines - 1; ++i) {
        pos = content.rfind('\n');
        if (pos == std::string::npos) {
            // not enough lines
            history_N_lines.push_back(content);
            break;
        }
        // If newline is at the very end, we have a trailing empty line: erase and continue
        if (pos + 1 == content.size()) {
            content.erase(pos);  // remove trailing '\n'
            continue;
        }
        std::string line = content.substr(pos + 1);
        content.erase(pos);
        if (line.empty()) {
            continue;
        }

        history_N_lines.push_back(line);
    }

    std::string history_N_lines_str = "";
    std::string last_command;

    int i = 0;
    for (auto it : history_N_lines) {
        printf("i: %d : %s\n", i, it.c_str());
        if (i == 0) {
            // skip current command executing
            i++;
            continue;
        } else if (i == 1) {
            last_command = it;
        }
        history_N_lines_str += std::to_string(i) + ": " + it + "\n";
        i++;
    }
    last_command = "<last_command>\n" + last_command + "<last_command>\n";
    printf("last_command: %s\n", last_command.c_str());

    history_N_lines_str = "<history purpose='recent_commands' source='" + history_file.string() +
                          "' truncated='true'>>\n" + history_N_lines_str + "\n</history>\n";
    return last_command + history_N_lines_str;
}

static std::string get_system_context() {
    std::string sys_env = "<env>\n";
    sys_env += "cwd: " + std::filesystem::current_path().string() + "\n";
    sys_env += "OS: " + get_os() + "\n";
    sys_env += "shell:\n" + get_shell_name() + "\n";
    return sys_env + "\n</env>\n";
}

static int get_default_model(httplib::Client & cli, std::string & model_name) {
    const std::string model_endpoint = "/v1/models";
    try {
        auto res = cli.Get(model_endpoint);
        if (!res) {
            fprintf(stderr, "HTTP request failed %s\n", httplib::to_string(res.error()).c_str());
            return 1;
        }
        if (res->status != 200) {
            fprintf(stderr, "HTTP request returned status %d\n", res->status);
            return 1;
        }
        auto j = json::parse(res->body);
        if (j.contains("data") && j["data"].is_array() && !j["data"].empty() && j["data"][0].contains("id")) {
            model_name = j["data"][0]["id"].get<std::string>();
            return 0;
        } else {
            fprintf(stderr, "No models found in response\n");
            return 1;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

static int oai_send_prompt(httplib::Client & cli, json & payload, std::string & response) {
    std::string oai_endpoint = "/v1/chat/completions";
    try {
        const std::string content_type = "application/json";

        std::string json_payload = payload.dump();
        auto        res          = cli.Post(oai_endpoint, json_payload, content_type);
        if (!res) {
            fprintf(stderr, "HTTP request failed (%s) to %s\n", httplib::to_string(res.error()).c_str(),
                    oai_endpoint.c_str());
            return 1;
        }
        if (res->status == 200) {
            // Success
            if (res->body.empty()) {
                fprintf(stderr, "Empty response body\n");
                return 1;
            }
            json data = json::parse(res->body);
            response  = data["choices"][0]["message"]["content"].get<std::string>();
            return 0;
        } else {
            fprintf(stderr, "HTTP request to %s returned status %d: %s\n", oai_endpoint.c_str(), res->status,
                    res->reason.c_str());
            return 1;
        }

    } catch (const std::exception & e) {
        // Catch any JSON parsing or other exceptions
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

static int oai_stream_prompt(httplib::Client & cli, json & payload, std::string & response) {
    const std::string oai_endpoint = "/v1/chat/completions";
    std::string       buffer;  // Buffer to hold incoming data chunks

    httplib::ContentReceiver content_receiver = [&](const char * data, size_t data_length) {
        // This will be called as data chunks are received
        buffer.append(data, data_length);
        size_t pos;
        while ((pos = buffer.find("\n")) != std::string::npos) {
            std::string line = buffer.substr(0, pos);
            buffer.erase(0, pos + 1);                    // +1 to remove the newline character
            if (line.empty()) {
                continue;                                // Skip empty lines
            }
            if (line.rfind("data: ", 0) == 0) {          // Line starts with "data: "
                std::string json_data = line.substr(6);  // Remove "data: "

                if (json_data == "[DONE]") {
                    return true;  // End of stream
                }

                try {
                    json data = json::parse(json_data);
                    if (data["choices"][0]["delta"].contains("role")) {
                        // initial chunk with role info
                        continue;
                    } else if (!data["choices"][0]["finish_reason"].empty()) {
                        // last chunk
                        std::cout << "\n" << std::flush;
                        continue;
                    }
                    std::string content = data["choices"][0]["delta"]["content"].get<std::string>();
                    response += content;
                    std::cout << content << std::flush;  // Print incrementally
                } catch (const std::exception & e) {
                    fprintf(stderr, "content_receiver error parsing data: %s\n", e.what());
                    break;
                }
            }
        }
        return true;
    };

    try {
        const std::string content_type    = "application/json";
        httplib::Headers  default_headers = {
            { "User-Agent", "llama-copilot" }
        };

        std::string               json_payload = payload.dump();
        httplib::DownloadProgress progress     = nullptr;

        auto res = cli.Post(oai_endpoint, default_headers, json_payload, content_type, content_receiver, progress);
        if (!res) {
            fprintf(stderr, "HTTP request failed (%s) to %s\n", httplib::to_string(res.error()).c_str(),
                    oai_endpoint.c_str());
            return 1;
        }
        switch (res->status) {
            case 200:
                // Success
                break;
            case 400:
                fprintf(stderr, "ERROR: malformed request, exceed_context_size_error\n");
            default:
                fprintf(stderr, "HTTP request to %s returned status %d: %s\n", oai_endpoint.c_str(), res->status,
                        res->reason.c_str());
                return 1;
        }

    } catch (const std::exception & e) {
        // Catch any JSON parsing or other exceptions
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 1;
}


static int set_up_http_client(httplib::Client & cli) {
    try {
        // Optional timeouts
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(5, 0);
        cli.set_write_timeout(5, 0);
        cli.set_default_headers({
            { "User-Agent", "llama-copilot" }
        });
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

static int set_up_json_payload(json &              payload,
                               const std::string & model_name,
                               bool                use_stream,
                               int                 n_predict,
                               const std::string & prompt) {
    try {
        payload["model"]     = model_name;
        payload["messages"]  = json::array({
            { { "role", "system" }, { "content", sp } },
            { { "role", "user" },  { "content", prompt } }
        });
        payload["n_predict"] = n_predict;
        payload["stream"]    = use_stream;
        payload["cache_prompt"] = true;
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

static std::string get_context(const char * filename, int n_ctx) {
    // add system context to prompt
    bool success;
    std::string context = "";

    if (!ISATTY(FILENO(stdin))) {
        std::string stdin_context = read_from_stdin(success);
        if (!success) {
            fprintf(stderr, "Failed to read from stdin\n");
            exit(1);
        }
        context += stdin_context;
    }
    if(filename) {
        std::string context_file_str = read_from_file(filename, success);
        if (!success) {
            fprintf(stderr, "Failed to read prompt from file '%s'\n", filename);
            exit(1);
        }
        context_file_str = "<file_context truncated='true' filename='" + std::string(filename) + "'>\n" + context_file_str + "\n</file_context>\n";
        context += context_file_str;
    }


    context += get_terminal_history_N_lines(5);
    context += get_system_context();
    return context;
}

int main(int argc, char ** argv) {
    std::string      url;
    std::string      prompt;
    std::string      context_file_str;
    httplib::Headers default_headers = {
        { "User-Agent", "llama-copilot" }
    };
    char *context_filename = NULL;
    bool use_stream = false;
    int  n_predict  = 512;
    int  n_ctx      = 512;
    int  ret;
    // parse command line arguments
    {
        int i = 1;
        for (; i < argc; i++) {
            try {
                if (strcmp(argv[i], "-u") == 0) {
                    if (i + 1 < argc) {
                        url = argv[++i];
                    } else {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else if (strcmp(argv[i], "-n") == 0) {
                    if (i + 1 < argc) {
                        n_predict = std::stoi(argv[++i]);
                    } else {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else if (strcmp(argv[i], "-f") == 0) {
                    if (i + 1 < argc) {
                        context_filename = argv[++i];

                    } else {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--stream") == 0) {
                    use_stream = true;
                } else {
                    // prompt starts here
                    break;
                }
            } catch (std::exception & e) {
                fprintf(stderr, "error: %s\n", e.what());
                print_usage(argc, argv);
                return 1;
            }
        }
        if (url.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = "<user_prompt>\n" + std::string(argv[i++]);
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
            prompt += "\n</user_prompt>";
        }
    }
    prompt += get_context(context_filename, n_ctx);

    auto [cli, parts] = common_http_client(url);
    if (set_up_http_client(cli) != 0) {
        fprintf(stderr, "Failed to set up HTTP client\n");
        return 1;
    }

    std::string model_name;
    if ((ret = get_default_model(cli, model_name)) != 0) {
        fprintf(stderr, "Failed to get default model\n");
        return ret;
    }
    printf("Default model: %s\n", model_name.c_str());

    json payload;
    if ((ret = set_up_json_payload(payload, model_name, use_stream, n_predict, prompt)) != 0) {
        return ret;
    }

    std::string response;
    if (use_stream) {
        if ((ret = oai_stream_prompt(cli, payload, response)) != 0) {
            return ret;
        }
        return 0;
    } else {
        if ((ret = oai_send_prompt(cli, payload, response)) != 0) {
            return ret;
        }
        printf("Response: %s\n", response.c_str());
    }
    return 0;
}
