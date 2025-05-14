// In lstm_inference.cpp
int main(int argc, char* argv[]) {
    try {
        std::string input_file_path = "tree_representation.json";
        if (argc > 1) {
            input_file_path = argv[1];
        }
        std::cout << "Input file: " << input_file_path << std::endl;

        // Load metadata
        std::ifstream metadata_file("model_metadata.json");
        if (!metadata_file.is_open()) {
            throw std::runtime_error("Failed to open model_metadata.json");
        }
        json metadata;
        metadata_file >> metadata;
        int max_sequence_length = metadata["max_sequence_length"].get<int>();
        int seq_input_size = metadata["seq_input_size"].get<int>();
        int scalar_input_size = metadata["scalar_input_size"].get<int>();
        std::vector<std::string> node_features = metadata["node_features"].get<std::vector<std::string>>();
        std::vector<std::string> scalar_features = metadata["scalar_features"].get<std::vector<std::string>>();
        std::vector<std::string> skewed_features = metadata["skewed_features"].get<std::vector<std::string>>();
        std::vector<std::string> dropped_features = metadata["dropped_features"].get<std::vector<std::string>>();
        std::cout << "Metadata loaded: seq_input_size=" << seq_input_size << ", scalar_input_size=" << scalar_input_size << std::endl;

        // Load scalers
        RobustScaler scaler_node, scaler_scalar, scaler_y;
        scaler_node.load("scaler_node_params.json");
        scaler_scalar.load("scaler_scalar_params.json");
        scaler_y.load("scaler_y_params.json");
        if (scaler_node.center.size() != seq_input_size || scaler_scalar.center.size() != scalar_input_size) {
            throw std::runtime_error("Scaler dimensions do not match input sizes");
        }

        // Load JSON input
        std::ifstream input_file(input_file_path);
        if (!input_file.is_open()) {
            throw std::runtime_error("Failed to open " + input_file_path);
        }
        json json_data;
        input_file >> json_data;
        std::cout << "Input JSON loaded" << std::endl;

        // Extract features
        FeatureExtractor extractor;
        extractor.feature_names = node_features;
        extractor.skewed_features = skewed_features;
        extractor.dropped_features = dropped_features;
        auto features = extractor.extract(json_data);

        // Create sequence input
        std::vector<float> feature_vec;
        for (const auto& key : node_features) {
            feature_vec.push_back(features[key]);
        }
        auto scaled_features = scaler_node.transform(feature_vec);
        std::vector<float> seq_data(max_sequence_length * seq_input_size, 0.0f);
        for (int i = 0; i < max_sequence_length; ++i) {
            for (int j = 0; j < seq_input_size; ++j) {
                seq_data[i * seq_input_size + j] = scaled_features[j];
            }
        }

        // Create scalar input
        std::vector<float> scalar_vec;
        for (const auto& key : scalar_features) {
            if (std::find(dropped_features.begin(), dropped_features.end(), key) == dropped_features.end()) {
                scalar_vec.push_back(features[key]);
            }
        }
        auto scaled_scalar = scaler_scalar.transform(scalar_vec);

        // Determine device
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "Using CPU device" << std::endl;
        }

        // Load model
        torch::jit::script::Module model;
        try {
            model = torch::jit::load("model.pt");
            model.eval();
            model.to(device);
            std::cout << "Model loaded and moved to device" << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load model.pt: " + std::string(e.what()));
        }

        // Create tensors
        torch::Tensor seq_tensor = torch::from_blob(
            seq_data.data(),
            {1, max_sequence_length, seq_input_size},
            torch::kFloat
        ).clone().to(device);
        torch::Tensor scalar_tensor = torch::from_blob(
            scaled_scalar.data(),
            {1, scalar_input_size},
            torch::kFloat
        ).clone().to(device);

        // Run inference
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
        auto output = model.forward(inputs).toTensor();
        float scaled_output = output.item<float>();

        // Inverse transform output
        float log_output = scaler_y.inverse_transform(scaled_output, 0);
        float execution_time_ms = std::expm1(log_output);
        execution_time_ms = std::max(0.0f, execution_time_ms);

        std::cout << "Predicted execution time: " << execution_time_ms << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
