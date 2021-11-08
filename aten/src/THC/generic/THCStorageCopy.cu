#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorageCopy.cu"
#else

// conversions are delegated to THCTensor implementation
#define THC_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC, TYPECUDA)              \
  void THCStorage_(copyCuda##TYPEC)(                                  \
      THCState * state,                                               \
      THCStorage * self,                                              \
      struct THCuda##TYPECUDA##Storage * src) {                       \
    size_t self_numel = self->nbytes() / sizeof(scalar_t);            \
    size_t src_numel =                                                \
        src->nbytes() / THCuda##TYPECUDA##Storage_elementSize(state); \
    THArgCheck(self_numel == src_numel, 2, "size does not match");    \
    at::Tensor selfTensor = tensor_reclaim(                           \
        THCTensor_(newWithStorage1d)(state, self, 0, self_numel, 1)); \
    at::Tensor srcTensor = tensor_reclaim(                            \
        THCuda##TYPECUDA##Tensor_newWithStorage1d(                    \
            state, src, 0, src_numel, 1));                            \
    selfTensor.copy_(srcTensor);                                      \
  }

THC_CUDA_STORAGE_IMPLEMENT_COPY(Byte,Byte)

#undef THC_CUDA_STORAGE_IMPLEMENT_COPY

void THCStorage_(copyCuda)(THCState *state, THCStorage *self, THCStorage *src)
{
  THCStorage_(TH_CONCAT_2(copyCuda, Real))(state, self, src);
}

void THCStorage_(copy)(THCState *state, THCStorage *self, THCStorage *src)
{
  THCStorage_(copyCuda)(state, self, src);
}

#endif
