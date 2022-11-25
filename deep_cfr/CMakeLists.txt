find_package(OpenMP REQUIRED)


set(TENSORFLOW_DIR ../third_party/tensorflow)
# link dynamic tensorflow lib.
include_directories( 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TENSORFLOW_DIR}/include
  ${TENSORFLOW_DIR}/bazel_include
  ${TENSORFLOW_DIR}/include/nsync/public
  ${TENSORFLOW_DIR}/include/com_google_protobuf/src
) 
link_directories(${TENSORFLOW_DIR}/bin)

add_library (deep_cfr_base OBJECT
  device_manager.h
  vpevaluator.h
  vpevaluator.cc
  vpnet.h
  vpnet.cc
  play.h
  play.cc
  local_best_response.h
  local_best_response.cc
  universal_poker_exploitability.h
  universal_poker_exploitability.cc
)
target_include_directories (deep_cfr_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(deep_cfr_base PUBLIC tensorflow_cc OpenMP::OpenMP_CXX)

# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-multiple-definition -Wl,--whole-archive")

# link_libraries(profiler)
# link_libraries(tcmalloc)

add_executable(vpnet_test vpnet_test.cc
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(vpnet_test deep_cfr_base)
add_test(vpnet_test vpnet_test)

add_executable(run_deep_cfr run_deep_cfr.cc deep_cfr.h deep_cfr.cc
    deep_escfr_solver.h deep_oscfr_solver.h deep_oscfr_solver.cc deep_escfr_solver.cc
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_deep_cfr deep_cfr_base)

add_executable(run_double_cfr run_deep_cfr.cc deep_cfr.h double_neural_cfr.cc 
    deep_escfr_solver.h deep_escfr_solver.cc deep_rscfr_solver.h deep_rscfr_solver.cc
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_double_cfr deep_cfr_base)

add_executable(run_sbcfr run_deep_cfr.cc  deep_cfr.h 
    deep_escfr_solver.h sbcfr.cc  sbcfr_solver.h sbcfr_solver.cc
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_sbcfr deep_cfr_base)

add_executable(run_sbescfr run_deep_cfr.cc  deep_cfr.h 
    deep_escfr_solver.h deep_sbescfr.cc  deep_sbescfr_solver.h deep_sbescfr_solver.cc
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_sbescfr deep_cfr_base)

add_executable(run_sbcfr1 run_deep_cfr.cc  deep_cfr.h 
    deep_escfr_solver.h sbcfr1.cc sbcfr1_solver.h sbcfr1_solver.cc
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_sbcfr1 deep_cfr_base)

add_executable(run_raw run_deep_cfr.cc deep_cfr.h 
    raw_sbcfr.cc raw_sbcfr_solver.h raw_sbcfr_solver.cc
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_raw deep_cfr_base)

add_executable(run_online_learning run_deep_cfr.cc deep_cfr.h 
    online_learning.cc online_learning_solver.h online_learning_solver.cc
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_online_learning deep_cfr_base)

add_executable(run_ossbcfr ossbcfr.cc run_deep_cfr.cc  deep_cfr.h 
    ossbcfr_solver.cc  ossbcfr_solver.h
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_ossbcfr deep_cfr_base)

add_executable(run_dream dream.cc run_deep_cfr.cc  deep_cfr.h 
    dream_solver.cc  dream_solver.h
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_dream deep_cfr_base)

add_executable(run_nfsp nfsp.cc run_deep_cfr.cc  deep_cfr.h 
    nfsp_solver.cc  nfsp_solver.h
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_nfsp deep_cfr_base)

if (${BUILD_WITH_ACPC})
    add_executable(universal_poker_exploitability_test universal_poker_exploitability_test.cc 
        $<TARGET_OBJECTS:tests>
        ${OPEN_SPIEL_OBJECTS})
    target_link_libraries(universal_poker_exploitability_test deep_cfr_base )
    add_test(universal_poker_exploitability_test universal_poker_exploitability_test)
endif()

# add_executable(multithread_eigen_test multithread_eigen_test.cc 
#     ${OPEN_SPIEL_OBJECTS})
# target_link_libraries(multithread_eigen_test PUBLIC OpenMP::OpenMP_CXX)
