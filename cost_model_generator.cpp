#include <torch/script.h>
#include <torch/torch.h>

#include "Halide.h"

using namespace Halide;

// Simplified cost model using the pre-trained SimpleLSTMModel for inference
class CostModel : public Generator<CostModel> {
protected:
    bool allow_out_of_order_inputs_and_outputs() const override {
        return true;
    }

public:
    using Input = GeneratorInput;
    using Output = GeneratorOutput;

    // Inputs (same as original)
    Input<int> num_stages{"num_stages", 1};
    Input<int> batch_size{"batch_size", 1};
    Input<int> num_cores{"num_cores", 1};
    Input<Buffer<float>> pipeline_features{"pipeline_features", 3};
    Input<Buffer<float>> schedule_features{"schedule_features", 3};

    // Output (same as original)
    Output<Buffer<float>> prediction_output{"prediction_output", 1};

    // PyTorch model
    std::shared_ptr<torch::jit::script::Module> pytorch_model;

    // Constructor to load the PyTorch model
    CostModel() {
        try {
            pytorch_model = torch::jit::load("/path/to/model.pt");
            pytorch_model->eval();
            pytorch_model->to(torch::kCPU);  // Use torch::kCUDA if GPU is available
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the PyTorch model: " << e.what() << std::endl;
            throw;
        }
    }

    void generate() {
        Var n("n"), w("w");

        // Step 1: Prepare inputs for the PyTorch model
        // Your SimpleLSTMModel expects:
        // - seq_input: [batch_size, sequence_length=3, seq_input_size=92]
        // - scalar_input: [batch_size, scalar_input_size=69]
        // We need to map Halide's pipeline_features and schedule_features to these inputs

        int batch_size_val = batch_size;
        int num_stages_val = num_stages;
        const int sequence_length = 3;  // As defined in your Python code
        const int seq_input_size = 92;  // Based on len(FIXED_FEATURES)
        const int scalar_input_size = 69;  // After preprocessing in your Python code

        // Step 2: Map Halide features to your model's FIXED_FEATURES
        // Define the features as in your Python code (simplified mapping)
        std::vector<float> seq_input_data(batch_size_val * sequence_length * seq_input_size, 0.0f);
        std::vector<float> scalar_input_data(batch_size_val * scalar_input_size, 0.0f);

        // Unpack schedule_features as in the original code
        Func schedule_features_func("schedule_features_func");
        schedule_features_func(n, w) = schedule_features(n, w, 0);  // Simplified for mapping
        int idx = 0;
        Expr num_realizations = schedule_features_func(n, idx++);
        Expr num_productions = schedule_features_func(n, idx++);
        Expr points_computed_total = schedule_features_func(n, idx++);
        Expr innermost_loop_extent = schedule_features_func(n, idx++);
        Expr inner_parallelism = schedule_features_func(n, idx++);
        Expr outer_parallelism = schedule_features_func(n, idx++);
        Expr bytes_at_realization = schedule_features_func(n, idx++);
        Expr bytes_at_production = schedule_features_func(n, idx++);
        Expr bytes_at_root = schedule_features_func(n, idx++);
        Expr unique_bytes_read_per_realization = schedule_features_func(n, idx++);
        Expr working_set = schedule_features_func(n, idx++);
        Expr vector_size = schedule_features_func(n, idx++);
        Expr num_vectors = schedule_features_func(n, idx++);
        Expr num_scalars = schedule_features_func(n, idx++);
        Expr bytes_at_task = schedule_features_func(n, idx++);
        Expr working_set_at_task = schedule_features_func(n, idx++);
        Expr working_set_at_production = schedule_features_func(n, idx++);
        Expr working_set_at_realization = schedule_features_func(n, idx++);
        Expr working_set_at_root = schedule_features_func(n, idx++);

        // Step 3: Fill seq_input_data (mimicking your Python preprocessing)
        for (int b = 0; b < batch_size_val; b++) {
            for (int t = 0; t < sequence_length; t++) {
                int offset = (b * sequence_length + t) * seq_input_size;
                // Map to FIXED_FEATURES (partial mapping for brevity)
                seq_input_data[offset + 0] = 0.0f;  // cache_hits (not available, set to 0)
                seq_input_data[offset + 1] = 0.0f;  // cache_misses
                seq_input_data[offset + 2] = 0.0f;  // execution_time_ms (not available)
                seq_input_data[offset + 3] = evaluate(num_realizations, {n, b});
                seq_input_data[offset + 4] = evaluate(num_productions, {n, b});
                seq_input_data[offset + 5] = evaluate(points_computed_total, {n, b});
                seq_input_data[offset + 6] = evaluate(innermost_loop_extent, {n, b});
                seq_input_data[offset + 7] = evaluate(inner_parallelism, {n, b});
                seq_input_data[offset + 8] = evaluate(outer_parallelism, {n, b});
                seq_input_data[offset + 9] = evaluate(bytes_at_realization, {n, b});
                seq_input_data[offset + 10] = evaluate(bytes_at_production, {n, b});
                // ... (map the rest of FIXED_FEATURES similarly)
                // Derived features
                float total_parallelism = evaluate(inner_parallelism, {n, b}) + evaluate(outer_parallelism, {n, b});
                seq_input_data[offset + 22] = total_parallelism;  // total_parallelism
                seq_input_data[offset + 23] = evaluate(num_realizations, {n, b}) + evaluate(num_productions, {n, b});  // scheduling_count
                // ... (continue mapping derived features)
            }
        }

        // Step 4: Fill scalar_input_data (after preprocessing)
        for (int b = 0; b < batch_size_val; b++) {
            int offset = b * scalar_input_size;
            // Apply preprocessing as in your Python code
            // Log transform skewed features (e.g., bytes_processing_rate)
            float bytes_at_realization_val = evaluate(bytes_at_realization, {n, b});
            float log_bytes_processing_rate = log1p(bytes_at_realization_val / 1.0f);  // Simplified
            // Map features (after dropping low-importance ones)
            scalar_input_data[offset + 0] = evaluate(bytes_at_task, {n, b});
            scalar_input_data[offset + 1] = evaluate(working_set_at_root, {n, b});
            scalar_input_data[offset + 2] = bytes_at_realization_val;
            scalar_input_data[offset + 3] = evaluate(unique_bytes_read_per_realization, {n, b});
            // ... (map the rest after dropping low-importance features)
            scalar_input_data[offset + 4] = log_bytes_processing_rate;
            // Apply RobustScaler (hardcode parameters or load from file)
            // For simplicity, assume scaling is done (you’d need to load scaler parameters)
        }

        // Step 5: Create PyTorch tensors
        torch::Tensor seq_input_tensor = torch::from_blob(seq_input_data.data(),
                                                         {batch_size_val, sequence_length, seq_input_size},
                                                         torch::kFloat32);
        torch::Tensor scalar_input_tensor = torch::from_blob(scalar_input_data.data(),
                                                            {batch_size_val, scalar_input_size},
                                                            torch::kFloat32);

        // Step 6: Run the PyTorch model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(seq_input_tensor);
        inputs.push_back(scalar_input_tensor);
        torch::Tensor output_tensor;
        {
            torch::NoGradGuard no_grad;
            output_tensor = pytorch_model->forward(inputs).toTensor();
        }

        // Step 7: Convert PyTorch output to Halide
        // Output is [batch_size, 1] (log-scaled runtime)
        auto output_accessor = output_tensor.accessor<float, 2>();
        Func predicted_log_runtime("predicted_log_runtime");
        predicted_log_runtime(n) = 0.0f;
        for (int b = 0; b < batch_size_val; b++) {
            predicted_log_runtime(b) = output_accessor[b][0];
        }

        // Step 8: Inverse transform the output (undo log scaling)
        // Your model outputs log1p(runtime), so apply expm1
        Func predicted_runtime("predicted_runtime");
        predicted_runtime(n) = expm1(predicted_log_runtime(n));

        // Step 9: Set the final output
        prediction_output(n) = predicted_runtime(n);

        // Step 10: Set estimates for autoscheduling (same as original)
        num_cores.set_estimate(32);
        batch_size.set_estimate(80);
        num_stages.set_estimate(13);
        prediction_output.set_estimates({{0, 80}});
        pipeline_features.set_estimates({{0, head1_w}, {0, head1_h}, {0, 13}});
        schedule_features.set_estimates({{0, 80}, {0, head2_w}, {0, 13}});

        // Step 11: Simplified scheduling for inference
        Var no;
        prediction_output.compute_root().split(n, no, n, 8).parallel(no);
        prediction_output.bound(n, 0, batch_size);
    }
};

HALIDE_REGISTER_GENERATOR(CostModel, cost_model);
