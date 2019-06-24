#ifndef DD_FC_H
#define DD_FC_H

#include "NvInfer.h"

#include "cudnn.h"
#include "kernel.h"
#include "plugin.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{


class InnerProductWithAxisPlugin : public IPluginV2Ext
{
public:
  // API constructor

  InnerProductWithAxisPlugin(const Weights* weights, const Weights* biases, int axis, int nbOutputChannels);

  // constructor for deserialization
  InnerProductWithAxisPlugin(const void* data, size_t length);


  //!
  //! \brief Return the plugin type. Should match the plugin name returned by the corresponding plugin creator
  // \see IPluginCreator::getPluginName()
  //!
  const char* getPluginType() const;

  //!
  //! \brief Return the plugin version. Should match the plugin version returned by the corresponding plugin creator
  // \see IPluginCreator::getPluginVersion()
  //!
  const char* getPluginVersion() const;

  //!
  //! \brief Get the number of outputs from the layer.
  //!
  //! \return The number of outputs.
  //!
  //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
  //!
  int getNbOutputs() const  { return 1; }

  //!
  //! \brief Get the dimension of an output tensor.
  //!
  //! \param index The index of the output tensor.
  //! \param inputs The input tensors.
  //! \param nbInputDims The number of input tensors.
  //!
  //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
  //!
  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims);

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
  bool supportsFormat(DataType type, PluginFormat format) const;


  //!
  //! \brief Initialize the layer for execution. This is called when the engine is created.
  //!
  //! \return 0 for success, else non-zero (which will cause engine termination).
  //!
  int initialize();

  //!
  //! \brief Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.
  //! \see initialize()
  //!
  void terminate();

  //!
  //! \brief Find the workspace size required by the layer.
  //!
  //! This function is called during engine startup, after initialize(). The workspace size returned should be sufficient for any
  //! batch size up to the maximum.
  //!
  //! \return The workspace size.
  //!
  size_t getWorkspaceSize(int maxBatchSize) const {return 0;}

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
  int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream);

  //!
  //! \brief Find the size of the serialization buffer required.
  //!
  //! \return The size of the serialization buffer.
  //!
  size_t getSerializationSize() const;

  //!
  //! \brief Serialize the layer.
  //!
  //! \param buffer A pointer to a buffer to serialize data. Size of buffer must be equal to value returned by getSerializationSize.
  //!
  //! \see getSerializationSize()
  //!
  void serialize(void* buffer) const;

  //!
  //! \brief Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
  //!
  void destroy();

  //!
  //! \brief Set the namespace that this plugin object belongs to. Ideally, all plugin
  //! objects from the same plugin library should have the same namespace.
  //!
  void setPluginNamespace(const char* pluginNamespace);

  //!
  //! \brief Return the namespace of the plugin object.
  //!
  const char* getPluginNamespace() const;


  //!
  //! \brief Return the DataType of the plugin output at the requested index.
  //! The default behavior should be to return the type of the first input, or DataType::kFLOAT if the layer has no inputs.
  //! The returned data type must have a format that is supported by the plugin.
  //! \see supportsFormat()
  //!
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const;

  //! \brief Return true if output tensor is broadcast across a batch.
  //!
  //! \param outputIndex The index of the output
  //! \param inputIsBroadcasted The ith element is true if the tensor for the ith input is broadcast across a batch.
  //! \param nbInputs The number of inputs
  //!
  //! The values in inputIsBroadcasted refer to broadcasting at the semantic level,
  //! i.e. are unaffected by whether method canBroadcastInputAcrossBatch requests
  //! physical replication of the values.
  //!
  virtual bool isOutputBroadcastAcrossBatch(int outputIndex,  const bool* inputIsBroadcasted,
                                            int nbInputs) const
  {
    return false;
  }


  //! \brief Return true if plugin can use input that is broadcast across batch without replication.
  //!
  //! \param inputIndex Index of input that could be broadcast.
  //!
  //! For each input whose tensor is semantically broadcast across a batch,
  //! TensorRT calls this method before calling configurePlugin.
  //! If canBroadcastInputAcrossBatch returns true, TensorRT will not replicate the input tensor;
  //! i.e., there will be a single copy that the plugin should share across the batch.
  //! If it returns false, TensorRT will replicate the input tensor
  //! so that it appears like a non-broadcasted tensor.
  //!
  //! This method is called only for inputs that can be broadcast.
  //!
  virtual bool canBroadcastInputAcrossBatch(int inputIndex) const
  {
    return false;
  }

  void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                       int nbOutputs, const DataType* inputTypes, const DataType* outputTypes,
                       const bool* inputIsBroadcast, const bool* outputIsBroadcast,
                       PluginFormat floatFormat, int maxBatchSize) ;
    //!
    //! \brief Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    //!
    //! \param cudnn The cudnn context handle of the execution context
    //! \param cublas The cublas context handle of the execution context
    //! \param allocator The allocator used by the execution context
    //!
    //! This function is called automatically for each plugin when a new execution context is created.
    //! If the plugin needs per-context resource, it can be allocated here.
    //! The plugin can also get context-owned CUDNN and CUBLAS context here.
    //!
    virtual void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) {}

    //!
    //! \brief Detach the plugin object from its execution context.
    //!
    //! This function is called automatically for each plugin when a execution context is destroyed.
    //! If the plugin owns per-context resource, it can be released here.
    //!
    virtual void detachFromContext() {}



    virtual ~InnerProductWithAxisPlugin() override = default;


  //!
  //! \brief Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with these parameters.
  //!
  IPluginV2Ext* clone() const _TENSORRT_OVERRIDE;


  int axis_;
  int nbOutputChannels_, dimsAfterAxis_;
  int dimsBeforeAxis_;
  Weights kernelWeights_, biasWeights_;

  DataType dataType_{DataType::kFLOAT};
  void* deviceKernel_{nullptr};
  void* deviceBias_{nullptr};

  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;
  cudnnTensorDescriptor_t srcDescriptor_, dstDescriptor_;

  std::string namespace_;

 private:
  size_t type2size(DataType type) const {return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half);}
    void* copyToDevice(const void* data, size_t count) const ;
    void convertAndCopyToDevice(void*& deviceWeights, const Weights& weights) const ;
    void convertAndCopyToBuffer(char*& buffer, const Weights& weights) const ;
    void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size);

};




 class InnerProductWithAxisPluginCreator : public BaseCreator
 {
 public:

   InnerProductWithAxisPluginCreator();
   //!
   //! \brief Return the plugin name.
   //!
   const char* getPluginName() const override;

    //!
    //! \brief Return the plugin version.
    //!
   const char* getPluginVersion() const override;

    //!
    //! \brief Return a list of fields that needs to be passed to createPlugin.
    //! \see PluginFieldCollection
    //!
   const PluginFieldCollection* getFieldNames() override;

    //!
    //! \brief Return a plugin object. Return nullptr in case of error.
    //!
   IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    //!
    //! \brief Called during deserialization of plugin layer. Return a plugin object.
    //!
   IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

   ~InnerProductWithAxisPluginCreator() {}

 private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif
