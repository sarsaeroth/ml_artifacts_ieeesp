#!/usr/bin/env bash
# Recompute the paper's numbers from data/, then run the attribution end to end on the shipped sample (skipped with --numbers-only).
# Each command is printed as "$ <command line>", followed by its result: one number, or a table. The program,
# the subcommand and the arguments say what is computed from which shipped file. Nothing is compared against
# expected values; read the results next to the paper. The output is also written to verify_output.txt.
# Every command runs even if an earlier one fails; the exit status is non-zero if any command failed.
# Usage: ./verify.sh [--numbers-only]
set -euo pipefail
cd "$(dirname "$0")"

# interpreter: the artifact's .venv if ./install.sh created it, else $PYTHON, else python3
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY="${PYTHON:-python3}"
fi
export PYTHONDONTWRITEBYTECODE=1
EIGHT=cashraven,ipidea,ipninja,luminati,monkeysocks,oxylabs,popa,traffmonetizer
failed=0

# print the command line (arguments quoted for the shell where needed), then run it; a failure is
# remembered and the list continues
run() {
  local line="$"
  local arg
  local quoted
  for arg in "$@"; do
    case "$arg" in
      "" | *[!A-Za-z0-9_./:=,@+-]*)
        quoted=${arg//"'"/"'\\''"}
        line="$line '$quoted'"
        ;;
      *) line="$line $arg" ;;
    esac
  done
  echo "$line"
  if ! "$@"; then
    failed=1
  fi
}

{
# Section 4: corpus (Table 1 and its caption, label sources, label validity)
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where-in "provider=$EIGHT" --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where provider=cashraven --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where provider=ipidea --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where provider=ipninja --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where provider=luminati --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where provider=monkeysocks --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where provider=oxylabs --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where provider=popa --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where provider=traffmonetizer --sep
run "$PY" scripts/query.py table 'data/results/dex_reuse_v3.json::per_family::*' --field share --percent --digits 1
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where-in "provider=$EIGHT" --where label_source=mi_network --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where-in "provider=$EIGHT" --where label_source=dex_pivot --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=dropped_unvalidated_pivot --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where-in "provider=$EIGHT" --where label_source=version_pivot --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where-in "provider=$EIGHT" --where label_source=c2_domain_pivot --sep
run "$PY" scripts/query.py count "data/manifests/manifest_v3.csv" --where status=kept --where provider=monkeysocks --distinct dex_group
run "$PY" scripts/query.py value data/results/label_validity_dex.json::full::n --sep
run "$PY" scripts/query.py diff "data/results/proxy_v3/v3/open_world.json::n_corpus" data/results/label_validity_dex.json::full::n --sep
run "$PY" scripts/query.py value data/results/label_validity_dex.json::minus_monkeysocks_dormant::dropped --sep
run "$PY" scripts/query.py value data/results/label_validity_dex.json::minus_monkeysocks_no_code::dropped --sep
run "$PY" scripts/query.py stat sum data/results/label_validity_dex.json::minus_monkeysocks_dormant::dropped data/results/label_validity_dex.json::minus_monkeysocks_no_code::dropped --sep
run "$PY" scripts/query.py value data/results/label_validity_dex.json::minus_ipidea_ninevpn_only::dropped --sep
run "$PY" scripts/query.py value data/results/label_validity_dex.json::minus_oxylabs_no_code::dropped --sep
run "$PY" scripts/query.py stat sum data/results/label_validity_dex.json::minus_monkeysocks_no_code::dropped data/results/label_validity_dex.json::minus_oxylabs_no_code::dropped --sep
run "$PY" scripts/query.py value data/results/label_validity_dex.json::minus_all::dropped --sep
run "$PY" scripts/query.py value data/results/label_validity_dex.json::full::per_family_n::monkeysocks --sep
run "$PY" scripts/query.py value data/results/label_validity_dex.json::minus_all::per_family_n::oxylabs --sep
run "$PY" scripts/query.py value data/results/label_validity_dex.json::full::macro_f1 --digits 3
run "$PY" scripts/query.py value data/results/label_validity_dex.json::minus_all::macro_f1 --digits 3
run "$PY" scripts/query.py value data/results/label_validity_neardup.json::full::macro_f1 --digits 3
run "$PY" scripts/query.py value data/results/label_validity_neardup.json::minus_all::macro_f1 --digits 3
run "$PY" scripts/query.py diff data/results/label_validity_neardup.json::full::macro_f1 data/results/label_validity_neardup.json::minus_monkeysocks_no_code::macro_f1 --digits 3

# Section 5.1: distinct method signatures in a provider's sampled applications
run "$PY" scripts/derive.py over-seeds median data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field n_methods --sep
run "$PY" scripts/derive.py over-seeds min data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field n_methods --sep
run "$PY" scripts/derive.py over-seeds max data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field n_methods --sep
run "$PY" scripts/derive.py over-seeds median data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field n_methods --over-providers max --in-millions --digits 1

# Section 6.1: classification baselines (dataset)
run "$PY" scripts/query.py value "data/results/proxy_v3/e2a_standardized.json::labeled::best_linear" --digits 3
run "$PY" scripts/query.py value "data/results/proxy_v3/v4/cert_drebin.json::drebin_code::rf::f1" --digits 3
run "$PY" scripts/query.py value "data/results/proxy_v3/e2a_standardized.json::mama_package::best_linear" --digits 3
run "$PY" scripts/query.py value "data/results/proxy_v3_gneardup_fcg/v4/cert_drebin.json::drebin_code::linear_svm::f1" --digits 3
run "$PY" scripts/query.py value "data/results/proxy_v3_gneardup_fcg/e2a_standardized.json::labeled::best_linear" --digits 3
run "$PY" scripts/query.py value "data/results/proxy_v3/v3/open_world.json::O2::linear::macro_f1" --digits 3
run "$PY" scripts/query.py value "data/results/proxy_v3/v3/open_world.json::O2::linear::sd" --digits 3

# Section 6.2: method rankings (Table 2), multiple-kernel baseline, allocation rules
run "$PY" scripts/derive.py over-seeds count data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field provenance::p@25 --provider luminati
run "$PY" scripts/derive.py over-seeds min data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field n_samples
run "$PY" scripts/derive.py over-seeds max data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field n_samples
run "$PY" scripts/derive.py over-seeds median data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field provenance::p@25 --field no_provenance::p@25 --field doc_freq::p@25 --field namespace::p@25 --field network_api::p@25 --field strings::p@25 --field random::p@25 --digits 2
run "$PY" scripts/derive.py over-seeds median data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field n_sdk --divide-by n_methods --digits 4
run "$PY" scripts/query.py table 'data/results/mkldroid.json::results::mkl::*' --skip-key beta --field p@25 --digits 2
run "$PY" scripts/query.py table 'data/results/mkldroid.json::results::identifier_only::*' --skip-key beta --field p@25 --digits 2
run "$PY" scripts/derive.py over-seeds median data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field random::p@25 --over-providers max --digits 2
run "$PY" scripts/derive.py over-seeds median data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field no_provenance::p@25 --over-providers max --digits 2
run "$PY" scripts/derive.py over-seeds min data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field provenance::p@25 --provider luminati --digits 2
run "$PY" scripts/derive.py over-seeds max data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field provenance::p@25 --provider luminati --digits 2
run "$PY" scripts/derive.py best-median-hits data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field provenance::p@25 --field no_provenance::p@25 --field doc_freq::p@25 --field namespace::p@25 --field network_api::p@25 --field strings::p@25 --field random::p@25 --top 25
run "$PY" scripts/derive.py best-median-hits data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field provenance::p@25 --field no_provenance::p@25 --field doc_freq::p@25 --field namespace::p@25 --field network_api::p@25 --field strings::p@25 --field random::p@25 --top 25 --hits-at-most 4
run "$PY" scripts/derive.py over-seeds median data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field provenance::auc --over-providers min --digits 2
run "$PY" scripts/derive.py over-seeds median data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field provenance::auc --over-providers max --digits 2
run "$PY" scripts/query.py value data/results/mkldroid.json::results::mkl::beta::identifier --digits 2
run "$PY" scripts/query.py value data/results/mkldroid.json::results::mkl::beta::fuzzy --digits 2
run "$PY" scripts/query.py value data/results/mkldroid.json::results::mkl::beta::api --digits 2
run "$PY" scripts/query.py value data/results/mkldroid.json::n_samples
run "$PY" scripts/query.py log 2 data/results/mkldroid.json::dim
run "$PY" scripts/query.py table 'data/results/attribution_robustness_v3.json::summary::*' --field exact::p_at_25 --field uncentered::p_at_25 --field out_of_fold::p_at_25 --field legacy::p_at_25 --digits 2
run "$PY" scripts/query.py table 'data/results/attribution_robustness_v3.json::summary::*' --field uncentered::overlap_top100 --field out_of_fold::overlap_top100 --field legacy::overlap_top100 --percent --digits 0
run "$PY" scripts/query.py count-equal 'data/results/attribution_robustness_v3.json::summary::*::uncentered::p_at_25' 'data/results/attribution_robustness_v3.json::summary::*::exact::p_at_25'
run "$PY" scripts/query.py stat min 'data/results/attribution_robustness_v3.json::summary::*::uncentered::overlap_top100' --percent --digits 0
run "$PY" scripts/query.py stat max 'data/results/attribution_robustness_v3.json::summary::*::uncentered::overlap_top100' --percent --digits 0
run "$PY" scripts/query.py count-equal 'data/results/attribution_robustness_v3.json::summary::*::out_of_fold::p_at_25' 'data/results/attribution_robustness_v3.json::summary::*::exact::p_at_25'
run "$PY" scripts/query.py stat min 'data/results/attribution_robustness_v3.json::summary::*::out_of_fold::overlap_top100' --percent --digits 0
run "$PY" scripts/query.py stat max 'data/results/attribution_robustness_v3.json::summary::*::out_of_fold::overlap_top100' --percent --digits 0
run "$PY" scripts/query.py stat min 'data/results/attribution_robustness_v3.json::summary::*::legacy::overlap_top100' --percent --digits 0
run "$PY" scripts/query.py stat max 'data/results/attribution_robustness_v3.json::summary::*::legacy::overlap_top100' --percent --digits 0
run "$PY" scripts/query.py count-equal 'data/results/attribution_robustness_v3.json::summary::*::legacy::p_at_25' 'data/results/attribution_robustness_v3.json::summary::*::exact::p_at_25'

# Section 6.2: deletion (Figure 2)
run "$PY" scripts/query.py value data/results/deletion_v3.json::summary::n
run "$PY" scripts/query.py table 'data/results/deletion_v3.json::summary::per_family::*' --field n
run "$PY" scripts/query.py stat max 'data/results/deletion_v3.json::summary::per_family::*::n'
run "$PY" scripts/query.py count 'data/results/deletion_v3.json::apps::*' --where family=traffmonetizer
run "$PY" scripts/query.py grid-table data/results/deletion_v3.json::p_grid data/results/deletion_v3.json::summary::attr data/results/deletion_v3.json::summary::rand --digits 3
run "$PY" scripts/query.py grid-table data/results/deletion_v3.json::p_grid 'data/results/deletion_v3.json::summary::per_family::*::attr' --digits 2
run "$PY" scripts/query.py value-at-grid-point data/results/deletion_v3.json::summary::attr --grid data/results/deletion_v3.json::p_grid --at 0 --digits 2
run "$PY" scripts/query.py value-at-grid-point data/results/deletion_v3.json::summary::attr --grid data/results/deletion_v3.json::p_grid --at 0.02 --digits 2

# Section 6.3: masking (Table 3), shares, renaming, combined coloring
run "$PY" scripts/query.py value data/results/confound_v3.json::n
run "$PY" scripts/query.py stat max 'data/results/confound_v3.json::counts::*'
run "$PY" scripts/query.py value data/results/confound_v3.json::dim
run "$PY" scripts/query.py one-over-count 'data/results/confound_v3.json::counts::*' --digits 3
run "$PY" scripts/query.py table 'data/results/confound_v3.json::conditions::*' --field macro_f1 --digits 3
run "$PY" scripts/query.py value data/results/confound_v3.json::conditions::full::macro_f1 --digits 2
run "$PY" scripts/query.py value data/results/confound_v3.json::conditions::sdk_masked::macro_f1 --digits 2
run "$PY" scripts/query.py table 'data/results/confound_v3.json::conditions::*' --field 'per_family_f1::*' --digits 2
run "$PY" scripts/query.py stat min 'data/results/confound_v3.json::conditions::sdk_only::per_family_f1::*' --digits 2
run "$PY" scripts/query.py stat max 'data/results/confound_v3.json::conditions::sdk_only::per_family_f1::*' --digits 2
run "$PY" scripts/query.py table 'data/results/confound_v3.json::share_median::*' --field sdk --field ad --percent --digits 1
run "$PY" scripts/query.py stat min 'data/results/confound_v3.json::share_median::*::sdk' --percent --digits 1
run "$PY" scripts/query.py stat max 'data/results/confound_v3.json::share_median::*::sdk' --percent --digits 1
run "$PY" scripts/query.py stat max 'data/results/confound_v3.json::share_median::*::ad' --percent --digits 1
run "$PY" scripts/query.py table 'data/results/masked_attribution_v3.json::summary::*' --field full::first_sdk_median --field ad_masked::first_sdk_median
run "$PY" scripts/query.py table 'data/results/masked_attribution_v3.json::summary::*' --field full::p_at_25 --field ad_masked::p_at_25 --field full::reach_within_3 --field ad_masked::reach_within_3 --digits 2
run "$PY" scripts/query.py value "data/results/proxy_v3/r3a_renaming.json::labeled::flip_rate" --percent --digits 1
run "$PY" scripts/query.py table "data/results/proxy_v3/r3a_renaming.json::labeled::per_family::*" --field flip_rate --percent --digits 0
run "$PY" scripts/query.py value "data/results/proxy_v3/e2a_standardized.json::combined::best_linear" --digits 3
run "$PY" scripts/query.py value "data/results/proxy_v3_gneardup/e2a_standardized.json::combined::best_linear" --digits 3
run "$PY" scripts/query.py value "data/results/proxy_v3_gneardup/e2a_standardized.json::labeled::best_linear" --digits 3

# Section 7.1: routes to the network core (Table 4), reach by hash dimension, diagnostic, bootstrap (Table 5)
run "$PY" scripts/derive.py over-seeds median data/results/analyst_heuristics.json::runs --seed-prefix v3/ --field provenance::p@25 --provider luminati --percent --digits 0
run "$PY" scripts/query.py table 'data/results/route_compare_v3.json::summary::*' --field explanation::within_3 --field explanation::reached --field marker::within_3 --field marker::reached --digits 2
run "$PY" scripts/query.py table 'data/results/route_compare_v3.json::summary::*' --field explanation::median_hops --field marker::median_hops
run "$PY" scripts/query.py stat max 'data/results/route_compare_v3.json::summary::*::explanation::within_3' --percent --digits 0
run "$PY" scripts/query.py table 'data/results/highdim_reach_v3.json::*' --field '*::p_at_25' --digits 2
run "$PY" scripts/query.py table 'data/results/highdim_reach_v3.json::*' --field '*::within_3' --percent --digits 0
run "$PY" scripts/query.py table 'data/results/explanation_diagnostic_v3.json::providers::*' --field sdk_share --percent --digits 1
run "$PY" scripts/query.py table 'data/results/explanation_diagnostic_v3.json::providers::*' --field sdk_only_f1 --field p_at_25 --field core_within_3 --field core_reached --digits 2
run "$PY" scripts/query.py count 'data/results/explanation_diagnostic_v3.json::providers::*' --above sdk_share=0.02
run "$PY" scripts/query.py count 'data/results/explanation_diagnostic_v3.json::providers::*' --above core_within_3=0
run "$PY" scripts/query.py table 'data/results/explanation_diagnostic_v3.json::spearman::*' --field rho --field p --digits 3
run "$PY" scripts/query.py value 'data/results/explanation_diagnostic_v3.json::spearman::sdk_share~core_within_3::rho' --digits 2
run "$PY" scripts/query.py value 'data/results/explanation_diagnostic_v3.json::spearman::sdk_share~core_within_3::p' --digits 3
run "$PY" scripts/query.py table 'data/results/bootstrap_chain_v3.json::summary::*' --field first_rank --field median_hops
run "$PY" scripts/query.py table 'data/results/bootstrap_chain_v3.json::summary::*' --field marker_coverage --field reached --field within_3 --digits 2
run "$PY" scripts/query.py table 'data/results/bootstrap_chain_v3.json::summary::*' --field markers
run "$PY" scripts/query.py table 'data/results/bootstrap_chain_v3.json::runs::*' --field '*::apps_scored'
run "$PY" scripts/query.py count 'data/results/bootstrap_chain_v3.json::summary::*' --above marker_coverage=0.5 --above reached=0

# Section 8.1 and Appendices A.2, A.5: minimum cut, SDK topology, reflection and native loading
run "$PY" scripts/query.py table 'data/results/mincut_v3.json::summary::*' --field analysed --field unambiguous --field cut_median --field sdk_methods_median --field sdk_wcc_median --field sdk_largest_scc_max
run "$PY" scripts/query.py table 'data/results/mincut_v3.json::summary::*' --field cut_zero_share --field pass_through_median --field sdk_reflection_share --field sdk_dex_loading_share --field sdk_native_loading_share --field host_reflection_share --field host_dex_loading_share --percent --digits 0
run "$PY" scripts/query.py stat sum 'data/results/mincut_v3.json::summary::*::analysed'
run "$PY" scripts/query.py stat max 'data/results/mincut_v3.json::summary::*::analysed'
run "$PY" scripts/query.py stat sum 'data/results/mincut_v3.json::summary::*::unambiguous'
run "$PY" scripts/query.py group-stat max 'data/results/mincut_v3.json::apps::*' --field cut --by family --nonempty cut
run "$PY" scripts/query.py group-stat max 'data/results/mincut_v3.json::apps::*' --field cut --by family --nonempty cut --over-groups max
run "$PY" scripts/query.py group-stat max 'data/results/mincut_v3.json::apps::*' --field cut --by family --nonempty cut --count-groups-at-max
run "$PY" scripts/query.py stat min 'data/results/mincut_v3.json::summary::*::sdk_methods_median' --digits 0 --sep
run "$PY" scripts/query.py stat max 'data/results/mincut_v3.json::summary::*::sdk_methods_median' --digits 0 --sep
run "$PY" scripts/query.py group-stat median 'data/results/mincut_v3.json::apps::*' --field nodes --by family --nonempty cut --sep
run "$PY" scripts/query.py group-stat median 'data/results/mincut_v3.json::apps::*' --field nodes --by family --nonempty cut --over-groups min --sep
run "$PY" scripts/query.py group-stat median 'data/results/mincut_v3.json::apps::*' --field nodes --by family --nonempty cut --over-groups max --sep
run "$PY" scripts/query.py count 'data/results/mincut_v3.json::apps::*' --where family=luminati --where cut=0
run "$PY" scripts/query.py stat min 'data/results/mincut_v3.json::summary::*::sdk_wcc_median' --digits 0
run "$PY" scripts/query.py stat max 'data/results/mincut_v3.json::summary::*::sdk_wcc_median' --digits 0
run "$PY" scripts/query.py stat max 'data/results/mincut_v3.json::summary::*::sdk_largest_scc_max'
run "$PY" scripts/query.py stat min 'data/results/mincut_v3.json::summary::*::pass_through_median' --percent --digits 0
run "$PY" scripts/query.py stat max 'data/results/mincut_v3.json::summary::*::pass_through_median' --percent --digits 0
run "$PY" scripts/query.py stat min data/results/mincut_v3.json::summary::luminati::sdk_reflection_share data/results/mincut_v3.json::summary::ipidea::sdk_reflection_share data/results/mincut_v3.json::summary::popa::sdk_reflection_share --percent --digits 0
run "$PY" scripts/query.py stat max data/results/mincut_v3.json::summary::luminati::sdk_reflection_share data/results/mincut_v3.json::summary::ipidea::sdk_reflection_share data/results/mincut_v3.json::summary::popa::sdk_reflection_share --percent --digits 0
run "$PY" scripts/query.py stat sum 'data/results/mincut_v3.json::summary::*::sdk_dex_loading_share'
run "$PY" scripts/query.py stat min 'data/results/mincut_v3.json::summary::*::host_reflection_share' --percent --digits 0
run "$PY" scripts/query.py stat min 'data/results/mincut_v3.json::summary::*::host_dex_loading_share' --percent --digits 0
run "$PY" scripts/query.py stat max 'data/results/mincut_v3.json::summary::*::host_dex_loading_share' --percent --digits 0

# Appendix A.1: manifest-declared SDK components (Table 7)
run "$PY" scripts/query.py table 'data/results/manifest_v3.json::summary::*' --field declared --field parsed --sep
run "$PY" scripts/query.py table 'data/results/manifest_v3.json::summary::*' --field rate --percent --digits 1

# Section 3 and Appendix A.6: search-space reduction
run "$PY" scripts/query.py value data/results/search_space_v3.json::top_fam
run "$PY" scripts/query.py stat max 'data/results/search_space_v3.json::per_fam_sampled::*'
run "$PY" scripts/query.py value data/results/search_space_v3.json::top_global
run "$PY" scripts/query.py value data/results/search_space_v3.json::ranked_before_dedup
run "$PY" scripts/query.py value data/results/search_space_v3.json::ranked_after_dedup
run "$PY" scripts/query.py value data/results/search_space_v3.json::entry_points
run "$PY" scripts/query.py value data/results/search_space_v3.json::entry_apps
run "$PY" scripts/query.py value data/results/search_space_v3.json::slice_methods
run "$PY" scripts/query.py value data/results/search_space_v3.json::slice_methods_nonlib

# Section 8.2 and Appendix A.7: token isolation against the e^{-M/d} model
run "$PY" scripts/query.py count data/results/exp_tier1_identifiability_truth_v2.jsonl
run "$PY" scripts/query.py table 'data/results/exp_tier1_identifiability_truth_v2.json::E1_isolation_vs_prediction::*' --field n
run "$PY" scripts/query.py table 'data/results/exp_tier1_identifiability_truth_v2.json::E1_isolation_vs_prediction::*' --field mean_signed_error --field median_signed_error --field max_abs_error --digits 5
run "$PY" scripts/query.py stat max-abs 'data/results/exp_tier1_identifiability_truth_v2.json::E1_isolation_vs_prediction::*::mean_signed_error' --digits 4
run "$PY" scripts/derive.py sdk-isolation-median data/results/exp_tier1_identifiability_truth_v2.jsonl --colouring labeled --field 'measured::*' --digits 3
run "$PY" scripts/derive.py sdk-isolation-median data/results/exp_tier1_identifiability_truth_v2.jsonl --colouring labeled --field 'predicted::*' --digits 3
run "$PY" scripts/derive.py sdk-isolation-median data/results/exp_tier1_identifiability_truth_v2.jsonl --colouring labeled --field measured::1048576 --over-providers min --digits 2
run "$PY" scripts/derive.py sdk-isolation-median data/results/exp_tier1_identifiability_truth_v2.jsonl --colouring labeled --field measured::1048576 --over-providers min --percent --digits 0
run "$PY" scripts/derive.py sdk-isolation-median data/results/exp_tier1_identifiability_truth_v2.jsonl --colouring labeled --field measured::1048576 --over-providers max --percent --digits 0
run "$PY" scripts/derive.py sdk-isolation-median data/results/exp_tier1_identifiability_truth_v2.jsonl --colouring labeled --field measured::1024 --count-below 0.0005
run "$PY" scripts/derive.py sdk-isolation-median data/results/exp_tier1_identifiability_truth_v2.jsonl --colouring labeled --field d_star_90 --log2 --digits 1
run "$PY" scripts/derive.py sdk-isolation-median data/results/exp_tier1_identifiability_truth_v2.jsonl --colouring labeled --field d_star_90 --log2 --over-providers max --digits 1
run "$PY" scripts/query.py value data/results/exp_tier1_identifiability_truth_v2.json::E4_colouring_vs_M_distinct::labeled::M_distinct_median --sep
run "$PY" scripts/query.py log 10 data/results/exp_tier1_identifiability_truth_v2.json::E4_colouring_vs_M_distinct::labeled::M_distinct_median --digits 0
run "$PY" scripts/query.py log e data/results/label_validity_dex.json::full::n --digits 0

# Appendix A.7: localization against hash dimension (Table 9)
run "$PY" scripts/query.py value data/results/highdim_localization.json::per_family_samples
run "$PY" scripts/query.py value data/results/highdim_localization.json::n_samples
run "$PY" scripts/query.py table 'data/results/highdim_localization.json::results::1024::families::*' --field p@100 --field p@25 --digits 2
run "$PY" scripts/query.py table 'data/results/highdim_localization.json::results::65536::families::*' --field p@100 --field p@25 --digits 2
run "$PY" scripts/query.py table 'data/results/highdim_localization.json::results::262144::families::*' --field p@100 --field p@25 --digits 2
run "$PY" scripts/query.py table 'data/results/highdim_localization.json::results::1048576::families::*' --field p@100 --field p@25 --digits 2
run "$PY" scripts/query.py table 'data/results/highdim_localization.json::results::1024::families::*' --field base_rate --digits 4
run "$PY" scripts/query.py table 'data/results/highdim_localization.json::results::*' --field macro_f1 --digits 3
run "$PY" scripts/query.py value data/results/highdim_localization.json::results::1024::families::oxylabs::sdk --sep

# Section 8.2 and Appendix A.8: dimension and holdout sweep (Figures 3 and 4)
run "$PY" scripts/query.py table 'data/results/sweep_v3.json::*' --digits 3

# attribution end to end on the shipped sample
if [ "${1:-}" != "--numbers-only" ]; then
  run "$PY" scripts/run_attribution.py --top 10
fi

exit "$failed"
} 2>&1 | tee verify_output.txt
