/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1
{
ILayer* parseLSTM(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    const trtcaffe::RecurrentParameter& p = msg.recurrent_param();

    int64_t seq_size = p.time_step();
    //int64_t nbInputs = parserutils::volume(parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions()));
    int64_t nbOutputs = p.num_output();

    // Weights kernelWeights = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) : weightFactory.allocateWeights(nbInputs * nbOutputs, std::normal_distribution<float>(0.0F, std_dev));
    // Weights biasWeights = !p.has_bias_term() || p.bias_term() ? (weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kBIAS) : weightFactory.allocateWeights(nbOutputs)) : weightFactory.getNullWeights();

    // weightFactory.convert(kernelWeights);
    // weightFactory.convert(biasWeights);

    std::cout << "LSTM seq_size:" << seq_size << std::endl;
    ILayer * lstm =network.addRNNv2(*tensors[msg.bottom(0)], 1, nbOutputs, seq_size, nvinfer1::RNNOperation::kLSTM);
    std::cout << "lstm input " << std::endl;
    ITensor * it = tensors[msg.bottom(0)];
    std::cout << " dims :" <<it->getDimensions().nbDims << std::endl;
    for (int  j=0; j< it->getDimensions().nbDims; ++j)
      std::cout <<  "     dim " << j <<  " : "  << it->getDimensions().d[j] << std::endl;
    for (int i=0; i< lstm->getNbOutputs(); ++i)
      {
        std::cout << "lstm output " << i << std::endl;
        ITensor * it = lstm->getOutput(i);
        std::cout << " dims :" <<it->getDimensions().nbDims << std::endl;
        for (int  j=0; j< it->getDimensions().nbDims; ++j)
          std::cout <<  "     dim " << j <<  " : "  << it->getDimensions().d[j] << std::endl;
      }
    return lstm;
    //return network.addFullyConnected(*tensors[msg.bottom(0)], p.num_output(), kernelWeights, biasWeights);
}
} //namespace nvcaffeparser1
