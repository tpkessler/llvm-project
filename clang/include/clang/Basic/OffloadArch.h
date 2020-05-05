//===--- OffloadArch.h - Utilities for offload arch -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OFFLOAD_ARCH_H
#define LLVM_CLANG_BASIC_OFFLOAD_ARCH_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace clang {

/// Get all feature strings that can be used in offload arch for \m Device.
/// Offload arch is a device name with optional offload arch feature strings
/// postfixed by a plus or minus sign delimited by colons, e.g.
/// gfx908:xnack+:sramecc-. Each device have limited
/// number of predefined offload arch features which have to follow predefined
/// order when showing up in a offload arch.
const llvm::SmallVector<llvm::StringRef, 4>
getAllPossibleOffloadArchFeatures(llvm::StringRef Device);

/// Parse an offload arch to get GPU arch and feature map.
/// Returns GPU arch.
/// Returns offload arch features in \p FeatureMap if it is not null pointer.
/// This function assumes \p OffloadArch is a valid offload arch.
/// If the offload arch contains feature+, map it to true.
/// If the offload arch contains feature-, map it to false.
/// If the offload arch does not contain a feature (default), do not map it.
/// Returns whether the offload arch features are valid in \p IsValid if it
/// is not a null pointer.
llvm::StringRef parseOffloadArch(llvm::StringRef OffloadArch,
                                 llvm::StringMap<bool> *FeatureMap = nullptr,
                                 bool *IsValid = nullptr);

} // namespace clang

#endif
