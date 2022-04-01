# Install script for directory: /Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos/xtflm_python.1.0.1.dylib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos" TYPE SHARED_LIBRARY FILES "/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/build/xtflm_python.1.0.1.dylib")
  if(EXISTS "$ENV{DESTDIR}/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos/xtflm_python.1.0.1.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos/xtflm_python.1.0.1.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -x "$ENV{DESTDIR}/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos/xtflm_python.1.0.1.dylib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos/xtflm_python.dylib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos" TYPE SHARED_LIBRARY FILES "/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/build/xtflm_python.dylib")
  if(EXISTS "$ENV{DESTDIR}/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos/xtflm_python.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos/xtflm_python.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -x "$ENV{DESTDIR}/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/libs/macos/xtflm_python.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/aranmcconnell/Documents/fix_ai_tools/ai_tools/third_party/lib_tflite_micro/xinterpreters/xinterpreters/host/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
