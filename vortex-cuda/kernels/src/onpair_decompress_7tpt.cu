// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#define TOKENS_PER_THREAD    7u
#define ONPAIR_KERNEL_NAME   onpair_decompress_7tpt
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#include "onpair_decompress_tpt.cuh"
