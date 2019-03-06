#include <algorithm>
//#include "common.h"
#include "tensorNet.h"
#include <sstream>
#include <fstream>

using namespace nvinfer1;


bool TensorNet::LoadNetwork(const char* prototxt_path,
                            const char* model_path,
                            const char* input_blob,
                            const std::vector<std::string>& output_blobs,
                            const char* prototxt_path2,
                            const char* model_path2,
                            const char* input_blob2,
                            const std::vector<std::string>& output_blobs2,
                            uint32_t maxBatchSize)
{
    //assert( !prototxt_path || !model_path );

    // attempt to load network from cache before profiling with tensorRT
    std::stringstream gieModelStdStream;
    std::stringstream gieModelStdStream2;
    gieModelStdStream.seekg(0, gieModelStdStream.beg);
    gieModelStdStream2.seekg(0, gieModelStdStream2.beg);
    char cache_path[512];
    sprintf(cache_path, "%s.%u.tensorcache", model_path, maxBatchSize);
    printf( "attempting to open cache file %s\n", cache_path);
    char cache_path2[512];
    sprintf(cache_path2, "%s.%u.tensorcache", model_path2, maxBatchSize);
    printf( "attempting to open cache file %s\n", cache_path2);

    std::ifstream cache( cache_path );
    std::ifstream cache2( cache_path2 );

    if( !cache || !cache2)
    {
        printf( "cache file not found, profiling network model\n");

        if( !caffeToTRTModel(prototxt_path, model_path, output_blobs, gieModelStdStream,prototxt_path2, model_path2, output_blobs2, gieModelStdStream2,maxBatchSize) )
        {
            printf("failed to load %s\n", model_path);
            return 0;
        }
        printf( "network profiling complete, writing cache to %s\n", cache_path);
        std::ofstream outFile;
        outFile.open(cache_path);
        outFile << gieModelStdStream.rdbuf();
        outFile.close();
        gieModelStdStream.seekg(0, gieModelStdStream.beg);
        printf( "completed writing cache to %s\n", cache_path);

        printf( "network profiling complete, writing cache to %s\n", cache_path2);
        std::ofstream outFile2;
        outFile2.open(cache_path2);
        outFile2 << gieModelStdStream2.rdbuf();
        outFile2.close();
        gieModelStdStream2.seekg(0, gieModelStdStream2.beg);
        printf( "completed writing cache to %s\n", cache_path2);

        infer = createInferRuntime(gLogger);
        infer2 = createInferRuntime(gLogger);
        /**
         * deserializeCudaEngine can be used to load the serialized CuDA Engine (Plan file).
         * */
        std::cout << "createInference" << std::endl;
        engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
        std::cout << "createInference_end" << std::endl;
        printf("Bindings after deserializing:\n");
        for (int bi = 0; bi < engine->getNbBindings(); bi++) {
            if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
            else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
        }

        std::cout << "createInference2" << std::endl;
        engine2 = infer2->deserializeCudaEngine(gieModelStream2->data(), gieModelStream2->size(), nullptr);
        std::cout << "createInference2_end" << std::endl;
        printf("Bindings after deserializing:\n");
        for (int bi = 0; bi < engine2->getNbBindings(); bi++) {
            if (engine2->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine2->getBindingName(bi));
            else printf("Binding %d (%s): Output.\n", bi, engine2->getBindingName(bi));
        }
    }
    else
    {
        std::cout << "loading network profile from cache..." << std::endl;
        gieModelStdStream << cache.rdbuf();
        cache.close();
        gieModelStdStream.seekg(0, std::ios::end);
        const int modelSize = gieModelStdStream.tellg();
        gieModelStdStream.seekg(0, std::ios::beg);
        void* modelMem = malloc(modelSize);
        gieModelStdStream.read((char*)modelMem, modelSize);

        infer = createInferRuntime(gLogger);
        std::cout << "createInference" << std::endl;
        engine = infer->deserializeCudaEngine(modelMem, modelSize, &pluginFactory);
        //free(modelMem);
        std::cout << "createInference_end" << std::endl;
        printf("Bindings after deserializing:\n");
        for (int bi = 0; bi < engine->getNbBindings(); bi++) {
            if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
            else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
        }

        std::cout << "loading network profile from cache2..." << std::endl;
        gieModelStdStream2 << cache2.rdbuf();
        cache2.close();
        gieModelStdStream2.seekg(0, std::ios::end);
        const int modelSize2 = gieModelStdStream2.tellg();
        gieModelStdStream2.seekg(0, std::ios::beg);
        void* modelMem2 = malloc(modelSize2);
        gieModelStdStream2.read((char*)modelMem2, modelSize2);

        infer2 = createInferRuntime(gLogger);
        std::cout << "createInference2" << std::endl;
        engine2 = infer2->deserializeCudaEngine(modelMem2, modelSize2, nullptr);
        //free(modelMem);
        std::cout << "createInference2_end" << std::endl;
        printf("Bindings after deserializing:\n");
        for (int bi = 0; bi < engine2->getNbBindings(); bi++) {
            if (engine2->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine2->getBindingName(bi));
            else printf("Binding %d (%s): Output.\n", bi, engine2->getBindingName(bi));
        }
    }
}

bool TensorNet::caffeToTRTModel(const char* deployFile,
                                const char* modelFile,
                                const std::vector<std::string>& outputs,
                                std::ostream& gieModelStdStream,
                                const char* deployFile2,
                                const char* modelFile2,
                                const std::vector<std::string>& outputs2,
                                std::ostream& gieModelStdStream2,
                                unsigned int maxBatchSize)
{
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilder* builder2 = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    INetworkDefinition *network2 = builder2->createNetwork();
    //    builder->setMinFindIterations(3);	// allow time for TX1 GPU to spin up
    //    builder->setAverageFindIterations(2);
    ICaffeParser* parser = createCaffeParser();
    ICaffeParser* parser2 = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);

    bool useFp16 = builder->platformHasFastFp16();

    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;

    std::cout << deployFile <<std::endl;
    std::cout << modelFile <<std::endl;
    // mobleNet_SSD

    const IBlobNameToTensor* blobNameToTensor =	parser->parse(deployFile,
                                                              modelFile,
                                                              *network,
                                                              modelDataType);
    assert(blobNameToTensor != nullptr);
    for (auto& s : outputs) network->markOutput(*blobNameToTensor->find(s.c_str()));

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 20);

    if(useFp16)
    {
        builder->setHalf2Mode(true);
    }
    ICudaEngine* engine = builder->buildCudaEngine( *network );
    assert(engine);
    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();
    if(!gieModelStream)
    {
        std::cout << "failed to serialize CUDA engine" << std::endl;
        return false;
    }
    gieModelStdStream.write((const char*)gieModelStream->data(),gieModelStream->size());


    // AlexNet
    std::cout << "Begin parsing model2..." << std::endl;
    const IBlobNameToTensor* blobNameToTensor2 = parser2->parse(deployFile2,
                                                              modelFile2,
                                                              *network2,
                                                              DataType::kHALF);
    if(useFp16)
    {
        builder2->setHalf2Mode(true);
    }
    std::cout << "End parsing model2..." << std::endl;
    // specify which tensors are outputs
    for (auto& s : outputs2)
        network2->markOutput(*blobNameToTensor2->find(s.c_str()));

    // Build the engine
    builder2->setMaxBatchSize(maxBatchSize);
    builder2->setMaxWorkspaceSize(16 << 20);	

    std::cout << "Begin building engine2..." << std::endl;
    ICudaEngine* engine2 = builder2->buildCudaEngine(*network2);
    assert(engine2);
    std::cout << "End building engine2..." << std::endl;

    network2->destroy();
    parser2->destroy();
    gieModelStream2 = engine2->serialize();
    if(!gieModelStream2)
    {
        std::cout << "failed to serialize CUDA engine2" << std::endl;
        return false;
    }
    gieModelStdStream2.write((const char*)gieModelStream2->data(),gieModelStream2->size());


    engine->destroy();
    builder->destroy();
    pluginFactory.destroyPlugin();
    engine2->destroy();
    builder2->destroy();
    shutdownProtobufLibrary();

    std::cout << "caffeToTRTModel Finished" << std::endl;
    return true;
}

/**
 * This function de-serializes the cuda engine.
 * */
void TensorNet::createInference()
{
    infer = createInferRuntime(gLogger);
    /**
     * deserializeCudaEngine can be used to load the serialized CuDA Engine (Plan file).
     * */
    engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

    printf("Bindings after deserializing:\n");
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
}

void TensorNet::imageInference(void** buffers, int nbBuffer, int batchSize)
{
    //std::cout << "Came into the image inference method here. "<<std::endl;
    assert( engine->getNbBindings()==nbBuffer);
    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);
    context->execute(batchSize, buffers);
    context->destroy();
}
// for Alexnet
// void TensorNet::imageInferenceForAlex(void** buffers, int nbBuffer, int batchSize)
// {
//     //std::cout << "Came into the image inference method here. "<<std::endl;
//     assert( engine2->getNbBindings()==nbBuffer);
//     IExecutionContext* context = engine->createExecutionContext();
//     context->setProfiler(&gProfiler);
//     context->execute(batchSize, buffers);
//     context->destroy();
// }

void TensorNet::timeInference(int iteration, int batchSize)
{
    int inputIdx = 0;
    size_t inputSize = 0;
    void* buffers[engine->getNbBindings()];

    for (int b = 0; b < engine->getNbBindings(); b++)
    {
        DimsCHW dims = static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
        size_t size = batchSize * dims.c() * dims.h() * dims.w() * sizeof(float);
        CHECK(cudaMalloc(&buffers[b], size));

        if(engine->bindingIsInput(b) == true)
        {
            inputIdx = b;
            inputSize = size;
        }
    }

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    CHECK(cudaMemset(buffers[inputIdx], 0, inputSize));

    for (int i = 0; i < iteration;i++) context->execute(batchSize, buffers);

    context->destroy();
    for (int b = 0; b < engine->getNbBindings(); b++) CHECK(cudaFree(buffers[b]));

}

DimsCHW TensorNet::getTensorDims(const char* name)
{
    for (int b = 0; b < engine->getNbBindings(); b++) {
        if( !strcmp( name, engine->getBindingName(b)) )
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
    }
    return DimsCHW{0,0,0};
}

DimsCHW TensorNet::getTensorDimsForAlex(const char* name)
{
    for (int b = 0; b < engine2->getNbBindings(); b++) {
        if( !strcmp( name, engine2->getBindingName(b)) )
            return static_cast<DimsCHW&&>(engine2->getBindingDimensions(b));
    }
    return DimsCHW{0,0,0};
}

//void TensorNet::getLayerOutput(void** buffers, int nbBuffer, int batchSize)
//{
//    /* *
//     * @TODO: Get the layer with name name in the network
//     * */
//    std::cout << "Came into the image inference method here. "<<std::endl;
//    assert( engine->getNbBindings()==nbBuffer);
//    IExecutionContext* context = engine->createExecutionContext();
//    context->setProfiler(&gProfiler);
//    context->execute( batchSize , buffers);
//
//    context->destroy();
//
//}

void TensorNet::printTimes(int iteration)
{
    gProfiler.printLayerTimes(iteration);
}

void TensorNet::destroy()
{
    pluginFactory.destroyPlugin();
    engine->destroy();
    infer->destroy();
}


bool SortByProb(const std::pair<int, float>& a,
                const std::pair<int, float>& b) {
  return (a.second > b.second);
}

void TensorNet::imageInferenceForAlex(void** buffers, int nbBuffer, int batchSize)
{
  assert(engine2->getNbBindings() == 2);
//   void *buffers[2];

  // 1.- Create an execution context to store intermediate activation values.
  IExecutionContext *context = engine2->createExecutionContext();
  context->setProfiler(&gProfiler);

  // 2.- Use the input and output layer names to get the correct input and
  // output indexes.
  int inputIndex = engine2->getBindingIndex("data");
  int outputIndex = engine2->getBindingIndex("prob");

  // allocate GPU buffers
  Dims3 inputDims =
            static_cast<Dims3 &&>(engine2->getBindingDimensions(inputIndex)),
        outputDims =
            static_cast<Dims3 &&>(engine2->getBindingDimensions(outputIndex));

  // Size in bytes of the input and output.
  // The input size is 618348 bytes, since: 1 x 227 x 227 x 3 x 4 = 618348.
  // Our batch size is one, the image is 227 pixels by 227 pixels in size,
  // there are 3 channels (BGR) and the size of a float is 4 bytes.
  size_t inputSize = batchSize * inputDims.d[0] * inputDims.d[1] *
                     inputDims.d[2] * sizeof(float);
  // The output size is 4000 since: 1 x 1000 x 1 x 1 x 4 = 4000 bytes.
  // Out batch size is one, we have 1000 probabilities in the output, and the
  // rest is 1.
  size_t outputSize = batchSize * outputDims.d[0] * outputDims.d[1] *
                      outputDims.d[2] * sizeof(float);
  if (1) {
    std::cout << "inputSize      : " << inputSize << std::endl;
    std::cout << "batchSize      : " << batchSize << std::endl;
    std::cout << "inputDims.d[0] : " << inputDims.d[0] << std::endl;
    std::cout << "inputDims.d[1] : " << inputDims.d[1] << std::endl;
    std::cout << "inputDims.d[2] : " << inputDims.d[2] << std::endl;
    std::cout << "sizeof(float)  : " << sizeof(float) << std::endl;

    std::cout << "outputIndex     : " << outputIndex << std::endl;
    std::cout << "outputSize     : " << outputSize << std::endl;
    std::cout << "batchSize      : " << batchSize << std::endl;
    std::cout << "outputDims.d[0]: " << outputDims.d[0] << std::endl;
    std::cout << "outputDims.d[1]: " << outputDims.d[1] << std::endl;
    std::cout << "outputDims.d[2]: " << outputDims.d[2] << std::endl;
    std::cout << "sizeof(float)  : " << sizeof(float) << std::endl;
  }
  // 3.- Using these indices, set up a buffer array pointing to the input
//   // and output buffers on the GPU.
//   CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
//   CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

//   CHECK(
//       cudaMemcpy(buffers[inputIndex], rgbData, inputSize, cudaMemcpyHostToDevice));

//   // 4.- Run the image through the network, while typically asynchronous, we
//   // will do synchronous in this case. We do it the number of times required
//   // to profile.
  for (int i = 0; i < 1; i++)
    context->execute(batchSize, buffers);

  float prob[2];
  CHECK(cudaMemcpy(prob, buffers[outputIndex], 2 * sizeof(float),
                   cudaMemcpyDeviceToHost));

  typedef std::pair<int, float> inference;
  std::vector<inference> results;
  for (int i = 0; i < 2; ++i) {
    results.push_back(std::make_pair(i, prob[i]));
  }
  std::sort(results.begin(), results.end(), SortByProb);

  for (int i = 0; i < 2; ++i) {
   int index = results.at(i).first;
    printf("inference result:%d %4.2f\n",index,
            results.at(i).second * 100);
}

// //  PrintInference(prob, LABELS_FILE, HOTDOG_MODE);
//   // Release the context and buffers
    context->destroy();
//   CHECK(cudaFree(buffers[inputIndex]));
//   CHECK(cudaFree(buffers[outputIndex]));
}