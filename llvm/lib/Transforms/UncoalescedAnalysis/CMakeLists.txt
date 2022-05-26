# If we don't need RTTI or EH, there's no reason to export anything
# from the hello plugin.
# if( NOT LLVM_REQUIRES_RTTI )
#   if( NOT LLVM_REQUIRES_EH )
#     set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/Hello.exports
#   endif()
# endif()

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

#set(CMAKE_BUILD_TYPE "Debug")
set(LLVM_ENABLE_PLUGINS ON)
set(LLVM_LINK_COMPONENTS
  Demangle
)

add_llvm_library(LLVMUncoalescedAnalysis MODULE
  InterprocUncoalescedAnalysisPass.cpp
  UncoalescedAnalysisPass.cpp
  GPUState.cpp
  MultiplierValue.cpp
  UncoalescedAnalysis.cpp

  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )
