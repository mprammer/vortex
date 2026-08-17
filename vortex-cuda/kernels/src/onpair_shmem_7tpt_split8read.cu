// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#define TOKENS_PER_THREAD    7u
#define ONPAIR_KERNEL_NAME   onpair_shmem_7tpt_split8read
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#include "onpair_split8read_tpt.cuh"
