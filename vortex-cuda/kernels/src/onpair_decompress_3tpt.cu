// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Coarsening rung K=3. The launch bounds are deliberately identical to every other rung
// so that K is the only variable across the ladder; a per-rung retune would confound the
// coarsening trade-off with a block-geometry choice.
#define TOKENS_PER_THREAD    3u
#define ONPAIR_KERNEL_NAME   onpair_decompress_3tpt
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#include "onpair_decompress_tpt.cuh"
