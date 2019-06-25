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
#include "half.h"

using namespace nvinfer1;

namespace nvcaffeparser1
{
ILayer* parseLSTM(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    const trtcaffe::RecurrentParameter& p = msg.recurrent_param();

    int64_t nout = p.num_output();
    int seq_size = tensors[msg.bottom(0)]->getDimensions().d[0];
    int isize = tensors[msg.bottom(0)]->getDimensions().d[1];


    std::cout << "input isize: " << isize << std::endl;
    std::cout << "input seq_size: " << seq_size << std::endl;


    std::vector<Weights> allLSTMWeights = weightFactory.getAllWeights(msg.name());
    std::cout << "number of lstm weights structs: " << allLSTMWeights.size() << std::endl;

    for (int i=0; i< allLSTMWeights.size(); ++i)
      {
        std::cout << "w" << i << " size: " << allLSTMWeights[i].count << std::endl;
      }


    IRNNv2Layer * lstm =network.addRNNv2(*tensors[msg.bottom(0)], 1, nout, seq_size, nvinfer1::RNNOperation::kLSTM);
    for (int i=0; i< lstm->getNbOutputs(); ++i)
      {
        std::cout << "lstm output " << i << std::endl;
        ITensor * it = lstm->getOutput(i);
        std::cout << " dims :" <<it->getDimensions().nbDims << std::endl;
        for (int  j=0; j< it->getDimensions().nbDims; ++j)
          std::cout <<  "     dim " << j <<  " : "  << it->getDimensions().d[j] << std::endl;
      }

    if (weightFactory.getDataType() == DataType::kHALF)
      {
        for (int i=0; i<allLSTMWeights.size(); ++i)
          weightFactory.convert(allLSTMWeights[i]);
        const float16 * winput = static_cast<const float16*>(allLSTMWeights[0].values);
        const float16 * bias_c_data = static_cast<const float16 *>(allLSTMWeights[1].values);
        const float16 * wrec = static_cast<const float16 *>(allLSTMWeights[2].values);
        std::vector<float16> zeros(nout, float16(0.0));
      }
    else
      {
        const float * winput = static_cast<const float*>(allLSTMWeights[0].values);
        const float * bias_c_data = static_cast<const float*>(allLSTMWeights[1].values);
        const float * wrec = static_cast<const float*>(allLSTMWeights[2].values);
        std::vector<float> zeros(nout, 0.0);

        //weight xc should be hidden x input
        // same order as ncnn
        // weight rc are hidden x hidden -> !! rc x hidden  (matrix first !! )
        // same as ncnn also

        nvinfer1::Weights iww{nvinfer1::DataType::kFLOAT,winput, nout * isize};
        nvinfer1::Weights iwr{nvinfer1::DataType::kFLOAT,wrec, nout * nout};

        nvinfer1::Weights ibw{nvinfer1::DataType::kFLOAT,bias_c_data,nout};
        nvinfer1::Weights ibr{nvinfer1::DataType::kFLOAT,zeros.data(),nout};

        nvinfer1::Weights fww{nvinfer1::DataType::kFLOAT, winput + nout*isize, nout * isize};
        nvinfer1::Weights fwr{nvinfer1::DataType::kFLOAT,wrec + nout*nout, nout * nout};

        nvinfer1::Weights fbw{nvinfer1::DataType::kFLOAT,bias_c_data + nout, nout};
        nvinfer1::Weights fbr{nvinfer1::DataType::kFLOAT,zeros.data(),nout};

        nvinfer1::Weights oww{nvinfer1::DataType::kFLOAT, winput + 2*nout*isize, nout * isize};
        nvinfer1::Weights owr{nvinfer1::DataType::kFLOAT,wrec + 2*nout*nout, nout * nout};

        nvinfer1::Weights obw{nvinfer1::DataType::kFLOAT,bias_c_data+ 2*nout, nout};
        nvinfer1::Weights obr{nvinfer1::DataType::kFLOAT,zeros.data(),nout};

        nvinfer1::Weights cww{nvinfer1::DataType::kFLOAT,winput + 3*nout*isize,nout * isize};
        nvinfer1::Weights cwr{nvinfer1::DataType::kFLOAT,wrec + 3*nout*nout, nout * nout};

        nvinfer1::Weights cbw{nvinfer1::DataType::kFLOAT,bias_c_data + 3*nout , nout};
        nvinfer1::Weights cbr{nvinfer1::DataType::kFLOAT,zeros.data(),nout};

        lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kINPUT, true, iww);
        lstm->setBiasForGate(0, nvinfer1::RNNGateType::kINPUT, true, ibw);
        lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kINPUT, false, iwr);
        lstm->setBiasForGate(0, nvinfer1::RNNGateType::kINPUT, false, ibr);

        lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kFORGET, true, fww);
        lstm->setBiasForGate(0, nvinfer1::RNNGateType::kFORGET, true, fbw);
        lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kFORGET, false, fwr);
        lstm->setBiasForGate(0, nvinfer1::RNNGateType::kFORGET, false, fbr);

        lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kOUTPUT, true, oww);
        lstm->setBiasForGate(0, nvinfer1::RNNGateType::kOUTPUT, true, obw);
        lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kOUTPUT, false, owr);
        lstm->setBiasForGate(0, nvinfer1::RNNGateType::kOUTPUT, false, obr);

        lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kCELL, true, cww);
        lstm->setBiasForGate(0, nvinfer1::RNNGateType::kCELL, true, cbw);
        lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kCELL, false, cwr);
        lstm->setBiasForGate(0, nvinfer1::RNNGateType::kCELL, false, cbr);
      }



    return lstm;
}
} //namespace nvcaffeparser1
