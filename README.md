# Deep_learning_library

To transpose a 2 dimensional tensor (Matrix):

    Tensor t(tensor_shape{{2,3}},tensor_data{{1,2,3,4,5,6}});
    std::cout << t({0,0}) << " " << t({0,1}) << " " << t({0,2}) << std::endl;
    std::cout << t({1,0}) << " " << t({1,1}) << " " << t({1,2}) << std::endl;

    std::cout << "\n";

    Tensor t1 = t.transpose();
    std::cout << t1({0,0}) << " " << t1({0,1}) << std::endl;
    std::cout << t1({1,0}) << " " << t1({1,1}) << std::endl;
    std::cout << t1({2,0}) << " " << t1({2,1}) << std::endl;


To matrix multiply 2 two dimensional Tensor (Matrix):

    Tensor t(tensor_shape{{3,3}},tensor_data{{1,2,3,4,5,6,7,8,9}});
    std::cout << t({0,0}) << " " << t({0,1}) << " " << t({0,2}) << std::endl;
    std::cout << t({1,0}) << " " << t({1,1}) << " " << t({1,2}) << std::endl;
    std::cout << t({2,0}) << " " << t({2,1}) << " " << t({2,2}) << std::endl;

    std::cout << "\n";

    Tensor t1(tensor_shape{{3,2}},tensor_data{{1,2,3,4,5,6}});
    std::cout << t1({0,0}) << " " << t1({0,1}) << std::endl;
    std::cout << t1({1,0}) << " " << t1({1,1}) << std::endl;
    std::cout << t1({2,0}) << " " << t1({2,1}) << std::endl;
    
    std::cout << "\n";

    Tensor t2 = t.matmul_2D(t1);
    std::cout << t2({0,0}) << " " << t2({0,1}) << std::endl;
    std::cout << t2({1,0}) << " " << t2({1,1}) << std::endl;
    std::cout << t2({2,0}) << " " << t2({2,1}) << std::endl;


Scalar Multiplication:

    Tensor t1(tensor_shape{{2,2}},tensor_data{{1,2,3,4}});

    std::cout << t1({0,0}) << " " << t1({0,1}) << std::endl;
    std::cout << t1({1,0}) << " " << t1({1,1}) << std::endl;

    t1 = t1 * 3.0f;

    std::cout << "\n";
    std::cout << t1({0,0}) << " " << t1({0,1}) << std::endl;
    std::cout << t1({1,0}) << " " << t1({1,1}) << std::endl;



