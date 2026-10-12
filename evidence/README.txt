Verification records for DASDG 1.1.0

accepted_candidate_summary.json: independent artifact audit of the 1.1.0rc1
server run, including 20 CPU and 17 unified/GPU checks and the original archive
identities. Tests used synthetic fixtures; these are not traffic benchmark scores.
local_CPU_core.json, local_integration.json, local_external_operator.json:
local candidate acceptance records. DGCRN operator checks used separate local
upstream files, which are not included in this source distribution.
final_release_install.json: final 1.1.0 version installation checks after build.
release_provenance.json in the repository root: candidate-to-release source
identity and the complete version/metadata transformation record.

GPU equivalence concerns a single recorded environment and epoch-boundary resume.
The evidence does not establish missing target label support, real nontraffic
accuracy, arbitrary cross-environment determinism, or a fresh install of CUDA
and every dependency. Historical v1.0.2 results have separate protocols and
remain under archival/. The public release/deposit identifiers are finalized
only after the author's publication step.
