//===--- OffloadArch.cpp - Utilities for parsing offload arch -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/OffloadArch.h"
#include "llvm/Support/raw_ostream.h"
namespace clang {

const llvm::SmallVector<llvm::StringRef, 4>
getAllPossibleOffloadArchFeatures(llvm::StringRef Device) {
  llvm::SmallVector<llvm::StringRef, 4> Ret;
  if (Device == "gfx902" || Device == "gfx908" || Device == "gfx909" ||
      Device.startswith("gfx10"))
    Ret.push_back("xnack");
  if (Device == "gfx906" || Device == "gfx908" || Device == "gfx909")
    Ret.push_back("sramecc");
  return Ret;
}

llvm::StringRef parseOffloadArch(llvm::StringRef OffloadArch,
                                 llvm::StringMap<bool> *FeatureMap,
                                 bool *IsValid) {
  llvm::StringRef ArchStr;
  auto SetValid = [&](bool Valid) {
    if (IsValid)
      *IsValid = Valid;
    return ArchStr;
  };

  auto Split = OffloadArch.split(':');
  ArchStr = Split.first;
  if (!FeatureMap && !IsValid)
    return ArchStr;

  auto Features = Split.second;
  if (Features.empty())
    return SetValid(true);

  auto AllFeatures = getAllPossibleOffloadArchFeatures(ArchStr);
  unsigned CurIndex = 0;
  while (!Features.empty()) {
    auto Splits = Features.split(':');
    auto Sign = Splits.first.back();
    auto Feature = Splits.first.drop_back();
    llvm::errs() << Feature << " " << Sign << " " << Splits.second << '\n';
    if (Sign != '+' && Sign != '-')
      return SetValid(false);
    bool IsOn = Sign == '+';
    for (; CurIndex < AllFeatures.size(); ++CurIndex) {
      if (Feature == AllFeatures[CurIndex]) {
        if (FeatureMap)
          (*FeatureMap)[Feature] = IsOn;
        break;
      }
    }
    if (CurIndex == AllFeatures.size())
      return SetValid(false);
    Features = Splits.second;
  }
  return SetValid(true);
};
} // namespace clang
