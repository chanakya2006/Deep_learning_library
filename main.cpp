#include<vector>
#include<iostream>

struct tensor_shape
{
    std::vector<int> shape;
};

struct tensor_data
{
    std::vector<float> data;
};



class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    Tensor(const tensor_shape & input_shape, const tensor_data & input_data = tensor_data{{}}) : shape(input_shape.shape), data(input_data.data){
        uint_least64_t expected_size = 1;
        for(auto dimension : shape) expected_size *= dimension;

        if(data.size()){
            if (data.size() != expected_size) throw std::runtime_error("Constructor : Mismatch in input shape and input data");
        }
        else{
            data.resize(expected_size,0.0f);
        }
    }

    Tensor(const std::vector<int> & input_shape, const std::vector<float> & input_data = {}) : shape(input_shape), data(input_data){
        uint_least64_t expected_size = 1;
        for(auto dimension : shape) expected_size *= dimension;

        if(data.size()){
            if (data.size() != expected_size) throw std::runtime_error("Constructor : Mismatch in input shape and input data");
        }
        else{
            data.resize(expected_size,0.0f);
        }
    }
    
    std::vector<int> get_shape(){
        return shape;
    }

    void set_data(const std::vector<float> & input_data){        
        if(input_data.size() != data.size()) throw std::runtime_error("set_data : Mismatch in input shape and input data");
        data = input_data;
    }

    Tensor matmul_2D(const Tensor& other) const {
        if ( (shape.size() != 2) || (other.shape.size() != 2)) throw std::runtime_error("matmul_2D : This function is to be only used when the tensors are 2 Dimensional.");
        if (shape[1] != other.shape[0]) throw std::runtime_error("matmul_2D : The Number of rows in the first tensor is not equal to the columns of the second tensor.");
        
        int matrix_1_columns = shape[0] , matrix_1_rows = shape[1] , matrix_2_rows = other.shape[1];
        std::vector<float> ans_data(matrix_1_columns*matrix_2_rows);
        // matrix_1 = m x n
        // matrix_2 = n x k

        for(int m = 0; m < matrix_1_columns; m++){
            for(int k = 0; k < matrix_2_rows; k++){
                float sum = 0;
                for(int n = 0; n < matrix_1_rows; n++){
                    sum += this->data[m*matrix_1_rows + n] * other.data[n*matrix_2_rows + k];
                }
                ans_data[m * matrix_2_rows + k] = sum;
            }
        }
        Tensor ans{ {shape[0],other.shape[1]} , ans_data};
        return ans;
    }

    Tensor transpose() const {
        if (shape.size() != 2) throw std::runtime_error("transpose : The tensor should be 2 dimensional.");
        
        int matrix_row = shape[1] , matrix_col = shape[0];
        std::vector<float> ans_data(matrix_col*matrix_row);
        // matrix = m x n

        for(int n = 0; n < matrix_row; n++){
            for(int m = 0; m < matrix_col; m++){
                ans_data[m + n * matrix_col] = data[n + m * matrix_row];
            }
        }

        Tensor ans{ {shape[1] , shape[0]} , ans_data};
        return ans;
    }

    // Converts N-dimensional indices to 1D index
    int flatten(const std::vector<int>& indices) const {
        if (indices.size() != shape.size())
            throw std::runtime_error("flatten : Index dimensions mismatch");

        int flat_index = 0;
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape[i] || indices[i] < 0)
                throw std::runtime_error("flatten : Index out of bounds");

            flat_index += indices[i] * stride;
            stride *= shape[i];
        }
        return flat_index;
    }

    // operator() for read/write access
    float& operator()(std::vector<int> idx) {		// Used to update the value
        return data[flatten(std::vector<int>(idx))];
    }

    float operator()(std::vector<int> idx) const {	// Returns Value 
        return data[flatten(std::vector<int>(idx))];
    }

    Tensor operator+(const Tensor& other) const {
        if (this->shape.size() != other.shape.size()) throw std::runtime_error("operator + : Index Dimensions Mismatch");

        for(uint_least64_t i = 0; i < this->shape.size(); i++){
            if (this->shape[i] != other.shape[i]) throw std::runtime_error("operator + : Tensors have same dimensionality but different Indexes");
        }

        Tensor ans(tensor_shape{this->shape});
        for(uint_least64_t i = 0; i < this->data.size(); i++){
            ans.data[i] = this->data[i] + other.data[i];
        }
        return ans;
    }

    Tensor operator-(const Tensor& other) const {
        if (this->shape.size() != other.shape.size()) throw std::runtime_error("operator - : Index Dimensions Mismatch");

        for(uint_least64_t i = 0; i < this->shape.size(); i++){
            if (this->shape[i] != other.shape[i]) throw std::runtime_error("operator - : Tensors have same dimensionality but different Indexes");
        }

        Tensor ans(tensor_shape{this->shape});
        for(uint_least64_t i = 0; i < this->data.size(); i++){
            ans.data[i] = this->data[i] - other.data[i];
        }
        return ans;
    }

    Tensor operator*(float Scalar_value){
        Tensor ans(tensor_shape{this->shape});
        for(uint_least64_t i = 0; i < this->data.size(); i++){
            ans.data[i] = this->data[i] * Scalar_value;    
        }
        return ans;
    }

    Tensor operator*(const Tensor& other){ // Element wise multiplication , for matrix multiplication use matmul
        if (this->shape.size() != other.shape.size()) throw std::runtime_error("operator * :Index Dimensions Mismatch");

        for(uint_least64_t i = 0; i < this->shape.size(); i++){
            if (this->shape[i] != other.shape[i]) throw std::runtime_error("operator * : Tensors have same dimensionality but different Indexes");
        }

        Tensor ans(tensor_shape{other.shape});
        for(uint_least64_t i = 0; i < this->data.size(); i++){
            ans.data[i] = this->data[i] * other.data[i];
        }
        return ans;
    }

    void print_shape(){
        for(int x : shape) {
            std::cout << x << " ";
        }
        std::cout << '\n';
    }
    
    ~Tensor(){ // destructor
    }   // free memory
};


int main(){

    return 0;
}