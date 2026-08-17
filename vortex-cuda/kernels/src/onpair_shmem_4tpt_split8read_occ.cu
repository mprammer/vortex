// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Four-TPT member of the controlled 256-thread / four-block launch family.
#define TOKENS_PER_THREAD    4u
#define ONPAIR_KERNEL_NAME   onpair_shmem_4tpt_split8read_occ
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#include "onpair_split8read_tpt.cuh"
