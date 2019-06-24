#include "NvInfer.h"
#include "plugin.h"
#include "innerProductWithAxisPlugin.h"
#include "fp16.h"
#include <cudnn.h>

using namespace nvinfer1;
using nvinfer1::plugin::InnerProductWithAxisPlugin;
using nvinfer1::plugin::InnerProductWithAxisPluginCreator;

namespace
{
static const char* InnerProductWithAxisPlugin_PLUGIN_VERSION{"1"};
static const char* InnerProductWithAxisPlugin_PLUGIN_NAME{"InnerProductWithAxis"};
}

PluginFieldCollection InnerProductWithAxisPluginCreator::mFC{};
std::vector<PluginField> InnerProductWithAxisPluginCreator::mPluginAttributes;


InnerProductWithAxisPlugin::InnerProductWithAxisPlugin(const Weights* weights, const Weights* biases, int axis, int nbOutputChannels) : axis_(axis), nbOutputChannels_(nbOutputChannels)
{
  kernelWeights_ = weights[0];
  assert(kernelWeights_.type == DataType::kFLOAT || kernelWeights_.type == DataType::kHALF);

  biasWeights_ = biases[0];
  std::cout << "biaswcount: " << biasWeights_.count << std::endl;
  std::cout << "nbout: " << nbOutputChannels << std::endl;

  assert(biasWeights_.count == 0 || biasWeights_.count == nbOutputChannels_);
  assert(biasWeights_.type == DataType::kFLOAT || biasWeights_.type == DataType::kHALF);

  kernelWeights_.values = malloc(kernelWeights_.count * type2size(kernelWeights_.type));
  memcpy(const_cast<void*>(kernelWeights_.values), weights[0].values,
         kernelWeights_.count * type2size(kernelWeights_.type));

  biasWeights_.values = malloc(biasWeights_.count * type2size(biasWeights_.type));
  memcpy(const_cast<void*>(biasWeights_.values), biases[0].values,
         biasWeights_.count * type2size(biasWeights_.type));

  dimsAfterAxis_ = int(weights[0].count / nbOutputChannels_);


  CHECK(cudnnCreate(&cudnn_)); // initialize cudnn and cublas
  CHECK(cublasCreate(&cublas_));
  CHECK(cudnnCreateTensorDescriptor(&srcDescriptor_));
  CHECK(cudnnCreateTensorDescriptor(&dstDescriptor_));
  if (kernelWeights_.values)
    convertAndCopyToDevice(deviceKernel_, kernelWeights_);
  if (biasWeights_.values)
    convertAndCopyToDevice(deviceBias_, biasWeights_);

}

InnerProductWithAxisPlugin::InnerProductWithAxisPlugin(const void* data, size_t length)
{
  const char *d = static_cast<const char*>(data), *a = d;
  dimsAfterAxis_ = read<int>(d);
  nbOutputChannels_ = read<int>(d);
  axis_ = read<int>(d);

  kernelWeights_.count = dimsAfterAxis_ * nbOutputChannels_;
  kernelWeights_.values = nullptr;

  biasWeights_.count = nbOutputChannels_;
  biasWeights_.values = nullptr;

  dataType_  = read<DataType>(d);


  deserializeToDevice(d, deviceKernel_, kernelWeights_.count * type2size(dataType_));
  deserializeToDevice(d, deviceBias_, biasWeights_.count * type2size(dataType_));
  assert(d == a + length);
}

//!
//! \brief Return the plugin type. Should match the plugin name returned by the corresponding plugin creator
// \see IPluginCreator::getPluginName()
//!
const char* InnerProductWithAxisPlugin::getPluginType() const
{
  return InnerProductWithAxisPlugin_PLUGIN_NAME;
}

    //!
    //! \brief Return the plugin version. Should match the plugin version returned by the corresponding plugin creator
    // \see IPluginCreator::getPluginVersion()
    //!
const char* InnerProductWithAxisPlugin::getPluginVersion() const
{
  return InnerProductWithAxisPlugin_PLUGIN_VERSION;
}


//!
//! \brief Get the dimension of an output tensor.
//!
//! \param index The index of the output tensor.
//! \param inputs The input tensors.
//! \param nbInputDims The number of input tensors.
//!
//! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
//!
Dims InnerProductWithAxisPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
  assert(index == 0 && nbInputDims == 1);
  int prod = 1;
  for (int i=axis_-1; i<inputs[0].nbDims; ++i)
    prod *= inputs[0].d[i];
  assert(dimsAfterAxis_ == prod);
  Dims d;
  d.nbDims = axis_;
  for (int i=0; i < axis_-1; ++i)
    d.d[i] = inputs[0].d[i];
  d.d[axis_-1] = nbOutputChannels_;
  return d;
}

//!
//! \brief Check format support.
//!
//! \param type DataType requested.
//! \param format PluginFormat requested.
//! \return true if the plugin supports the type-format combination.
//!
//! This function is called by the implementations of INetworkDefinition, IBuilder, and ICudaEngine.
//! In particular, it is called when creating an engine and when deserializing an engine.
//!
bool InnerProductWithAxisPlugin::supportsFormat(DataType type, PluginFormat format) const
{
  return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW;
}


//!
//! \brief Initialize the layer for execution. This is called when the engine is created.
//!
//! \return 0 for success, else non-zero (which will cause engine termination).
//!
int InnerProductWithAxisPlugin::initialize()
{

  // CHECK(cudnnCreate(&cudnn_)); // initialize cudnn and cublas
  // CHECK(cublasCreate(&cublas_));
  // CHECK(cudnnCreateTensorDescriptor(&srcDescriptor_)); // create cudnn tensor descriptors we need for bias addition
  // CHECK(cudnnCreateTensorDescriptor(&dstDescriptor_));
  // if (kernelWeights_.values)
  //   convertAndCopyToDevice(deviceKernel_, kernelWeights_);
  // if (biasWeights_.values)
  //   convertAndCopyToDevice(deviceBias_, biasWeights_);
  return STATUS_SUCCESS;
}

//!
//! \brief Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.
//! \see initialize()
//!
void InnerProductWithAxisPlugin::terminate()
{
  CHECK(cudnnDestroyTensorDescriptor(srcDescriptor_));
  CHECK(cudnnDestroyTensorDescriptor(dstDescriptor_));
  CHECK(cublasDestroy(cublas_));
  CHECK(cudnnDestroy(cudnn_));
  if (deviceKernel_)
    {
      cudaFree(deviceKernel_);
      deviceKernel_ = nullptr;
    }
  if (deviceBias_)
    {
      cudaFree(deviceBias_);
      deviceBias_ = nullptr;
    }
}



//!
//! \brief Execute the layer.
//!
//! \param batchSize The number of inputs in the batch.
//! \param inputs The memory for the input tensors.
//! \param outputs The memory for the output tensors.
//! \param workspace Workspace for execution.
//! \param stream The stream in which to execute the kernels.
//!
//! \return 0 for success, else non-zero (which will cause engine termination).
//!
int InnerProductWithAxisPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs,
            void* workspace, cudaStream_t stream)
{
  //TODO wrt axis !!!
  // if cuda kernel needed, use a function compiled separately from a .cu
  // if only cublas / cudnn function used, can be done right here
  float onef{1.0f}, zerof{0.0f};
  __half oneh = fp16::__float2half(1.0f), zeroh = fp16::__float2half(0.0f);

  cublasSetStream(cublas_, stream);
  cudnnSetStream(cudnn_, stream);

  if (dataType_ == DataType::kFLOAT)
    {
      CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, dimsBeforeAxis_, nbOutputChannels_, dimsAfterAxis_, &onef,
                        reinterpret_cast<const float*>(inputs[0]), dimsAfterAxis_,
                        reinterpret_cast<const float*>(deviceKernel_), nbOutputChannels_,
                        &zerof,
                        reinterpret_cast<float*>(outputs[0]), dimsAfterAxis_));
      // sampleplugin does the transpose operation, and thus have a transposed result ...
      // CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, nbOutputChannels_,dimsBeforeAxis_, dimsAfterAxis_, &onef,
      //                   reinterpret_cast<const float*>(mDeviceKernel), dimsAfterAxis_,
      //                   reinterpret_cast<const float*>(inputs[0]), dimsAfterAxis_, &zerof,
      //                   reinterpret_cast<float*>(outputs[0]), nbOutputChannels_));
    }
  else
    {
      CHECK(cublasHgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, dimsBeforeAxis_, nbOutputChannels_, dimsAfterAxis_, &oneh,
                        reinterpret_cast<const __half*>(inputs[0]), dimsAfterAxis_,
                        reinterpret_cast<const __half*>(deviceKernel_), nbOutputChannels_,
                        &zeroh,
                        reinterpret_cast<__half*>(outputs[0]), dimsAfterAxis_));
    }
  // but sampleplugin uses results as if not transposed ...
  if (biasWeights_.count)
    {
      cudnnDataType_t cudnnDT = dataType_ == DataType::kFLOAT ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
      CHECK(cudnnSetTensor4dDescriptor(srcDescriptor_, CUDNN_TENSOR_NCHW, cudnnDT, 1, nbOutputChannels_, 1, 1));
      CHECK(cudnnSetTensor4dDescriptor(dstDescriptor_, CUDNN_TENSOR_NCHW, cudnnDT, dimsBeforeAxis_, nbOutputChannels_, 1, 1));
      CHECK(cudnnAddTensor(cudnn_, &onef, srcDescriptor_, deviceBias_, &onef, dstDescriptor_, outputs[0]));
    }

  return 0;

}

//!
//! \brief Find the size of the serialization buffer required.
//!
//! \return The size of the serialization buffer.
//!
size_t InnerProductWithAxisPlugin::getSerializationSize() const
{
  return sizeof(dimsAfterAxis_) + sizeof(nbOutputChannels_) + sizeof(axis_) + sizeof(dataType_) + (kernelWeights_.count + biasWeights_.count) * type2size(dataType_);

}

//!
//! \brief Serialize the layer.
//!
//! \param buffer A pointer to a buffer to serialize data. Size of buffer must be equal to value returned by getSerializationSize.
//!
//! \see getSerializationSize()
//!
void InnerProductWithAxisPlugin::serialize(void* buffer) const
{
  char *d = static_cast<char*>(buffer), *a = d;

  write(d, dimsAfterAxis_);
  write(d, nbOutputChannels_);
  write(d, axis_);
  write(d, dataType_);
  convertAndCopyToBuffer(d, kernelWeights_);
  convertAndCopyToBuffer(d, biasWeights_);
  assert(d == a + getSerializationSize());
}

//!
//! \brief Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
//!
void InnerProductWithAxisPlugin::destroy()
{
  delete this;
}

//!
//! \brief Set the namespace that this plugin object belongs to. Ideally, all plugin
//! objects from the same plugin library should have the same namespace.
//!
void InnerProductWithAxisPlugin::setPluginNamespace(const char* pluginNamespace)
{
  namespace_ = pluginNamespace;
}

//!
//! \brief Return the namespace of the plugin object.
//!
const char* InnerProductWithAxisPlugin::getPluginNamespace() const
{
  return namespace_.c_str();
}



//!
//! \brief Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with these parameters.
//!

IPluginV2Ext* InnerProductWithAxisPlugin::clone() const
{
  auto* plugin = new InnerProductWithAxisPlugin(&kernelWeights_, &biasWeights_ , axis_, nbOutputChannels_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void InnerProductWithAxisPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                     int nbOutputs, const DataType* inputTypes, const DataType* outputTypes,
                     const bool* inputIsBroadcast, const bool* outputIsBroadcast,
                     PluginFormat floatFormat, int maxBatchSize)
{
  ASSERT(nbInputs > axis_);
  int dimsBeforeAxis_ = 1;
  for (int i=0; i<axis_; ++i)
    dimsBeforeAxis_ *= inputDims->d[i];
  return;
}

nvinfer1::DataType InnerProductWithAxisPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
}



void* InnerProductWithAxisPlugin::copyToDevice(const void* data, size_t count) const
{
  void* deviceData;
  CHECK(cudaMalloc(&deviceData, count));
  CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
  return deviceData;
}

void InnerProductWithAxisPlugin::convertAndCopyToDevice(void*& deviceWeights, const Weights& weights) const
{
  if (weights.type != dataType_) // Weights are converted in host memory first, if the type does not match
    {
      size_t size = weights.count * (dataType_ == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
      void* buffer = malloc(size);
      for (int64_t v = 0; v < weights.count; ++v)
        if (dataType_ == DataType::kFLOAT)
          static_cast<float*>(buffer)[v] = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
        else
          static_cast<__half*>(buffer)[v] = fp16::__float2half(static_cast<const float*>(weights.values)[v]);

      deviceWeights = copyToDevice(buffer, size);
      free(buffer);
    }
  else
    deviceWeights = copyToDevice(weights.values, weights.count * type2size(dataType_));
}

void InnerProductWithAxisPlugin::convertAndCopyToBuffer(char*& buffer, const Weights& weights) const
{
  if (weights.type != dataType_)
    for (int64_t v = 0; v < weights.count; ++v)
      if (dataType_ == DataType::kFLOAT)
        reinterpret_cast<float*>(buffer)[v] = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
      else
        reinterpret_cast<__half*>(buffer)[v] = fp16::__float2half(static_cast<const float*>(weights.values)[v]);
  else
    memcpy(buffer, weights.values, weights.count * type2size(dataType_));
  buffer += weights.count * type2size(dataType_);
}

void InnerProductWithAxisPlugin::deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
{
  deviceWeights = copyToDevice(hostBuffer, size);
  hostBuffer += size;
}



InnerProductWithAxisPluginCreator::InnerProductWithAxisPluginCreator()
{
  mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("nbout", nullptr, PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* InnerProductWithAxisPluginCreator::getPluginName() const
{
  return InnerProductWithAxisPlugin_PLUGIN_NAME;
}

const char* InnerProductWithAxisPluginCreator::getPluginVersion() const
{
  return InnerProductWithAxisPlugin_PLUGIN_VERSION;
}

const PluginFieldCollection* InnerProductWithAxisPluginCreator::getFieldNames()
{
  return &mFC;
}


IPluginV2Ext* InnerProductWithAxisPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
  std::vector<float> weightValues;
  std::vector<float> biasesValues;
  const PluginField* fields = fc->fields;

  int  axis, nbout;


  for (int i = 0; i < fc->nbFields; ++i)
    {
      const char* attrName = fields[i].name;
      if (!strcmp(attrName, "axis"))
        {
          ASSERT(fields[i].type == PluginFieldType::kINT32);
          axis = *(static_cast<const int*>(fields[i].data));
        }
      else if (!strcmp(attrName, "nbout"))
        {
          ASSERT(fields[i].type == PluginFieldType::kINT32);
          nbout = *(static_cast<const int*>(fields[i].data));
        }
      else if (!strcmp(attrName, "weights"))
        {
          ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
          int size = fields[i].length;
          weightValues.reserve(size);
          const auto* w = static_cast<const float*>(fields[i].data);
          for (int j = 0; j < size; j++)
            {
              weightValues.push_back(*w);
              w++;
            }
        }
      else if (!strcmp(attrName, "biases"))
        {
          ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
          int size = fields[i].length;
          biasesValues.reserve(size);
          const auto* w = static_cast<const float*>(fields[i].data);
          for (int j = 0; j < size; j++)
            {
              biasesValues.push_back(*w);
              w++;
            }
        }

    }
  Weights weights{DataType::kFLOAT, weightValues.data(), (int64_t) weightValues.size()};
  Weights biases{DataType::kFLOAT, biasesValues.data(), (int64_t) biasesValues.size()};

  auto* plugin = new InnerProductWithAxisPlugin(&weights, &biases, axis, nbout);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

IPluginV2Ext* InnerProductWithAxisPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
  // This object will be deleted when the network is destroyed, which will
  // call Concat::destroy()
  IPluginV2Ext* plugin = new InnerProductWithAxisPlugin(serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}
