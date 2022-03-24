def date_str_bld = new Date().format("yyyy-mm-dd")

@Library('xmos_jenkins_shared_library@v0.16.3') _
getApproval()
pipeline {
    agent {
        dockerfile true 
    }
    stages {
        stage("Checkout repo") {
            steps {
                // clean auto default checkout
                sh "rm -rf *"
                // clone
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: false,
                    extensions: [[$class: 'SubmoduleOption',
                                  threads: 8,
                                  timeout: 20,
                                  shallow: true,
                                  parentCredentials: true,
                                  recursiveSubmodules: true],
                                 [$class: 'CleanCheckout']],
                    userRemoteConfigs: [[credentialsId: 'xmos-bot',
                                         url: 'git@github.com:xmos/lib_tflite_micro']]
                ])
                sh "conda -V"
                sh "conda env create -q -p lib_tflite_micro_venv -f ./environment.yml"
                sh """. activate ./lib_tflite_micro_venv &&
                pip3 install -r ./requirements.txt
                """                     
            }
            post {
                cleanup {
                    deleteDir()
                }
            }
        }
/*        stage("Cleanup2") {
            steps {
                // The Jenkins command deleteDir() doesn't seem very reliable, so we're using the basic form
//                sh("rm -rf *")
            }
        }*/
        stage("Checkout") {
            steps {
                dir("sb") {
                    unstash 'lib_tflite_micro'
                    sh 'git submodule update --depth=1 --init --recursive --jobs 8'
                    sh 'make init'
                }
            }
        }
        stage("Build") {
            steps {
                dir("sb") {
                    sh 'make build'
                }
            }
        }
        stage("Test") {
            steps {
                dir("sb") {
                    sh 'make test'
                }
            }
        }
    }
}
