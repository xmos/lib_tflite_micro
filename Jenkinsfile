def date_str_bld = new Date().format("yyyy-mm-dd")

@Library('xmos_jenkins_shared_library@v0.16.3') _
getApproval()
pipeline {
    agent {
        dockerfile true
    }
    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }
    stages {
        stage("Setup") {
            steps {
                sh "rm -rf *"

                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: false,
                    extensions: [[$class: 'SubmoduleOption',
                                threads: 8,
                                timeout: 20,
                                shallow: false,
                                parentCredentials: true,
                                recursiveSubmodules: true],
                                [$class: 'CleanCheckout']],
                    userRemoteConfigs: [[credentialsId: 'xmos-bot',
                                        url: 'git@github.com:xmos/lib_tflite_micro']]
                ])
                // create .venv folder
                sh "conda -V"
                sh "conda env create -q -p .venv -f environment.yml"

                }
            }
        stage("Update environment") {
            steps {
                sh "conda update --all -y -q -p .venv"
                sh ". activate ./.venv"
                sh "make init"                     
            }
        }
/*        stage("Cleanup2") {
            steps {
                // The Jenkins command deleteDir() doesn't seem very reliable, so we're using the basic form
//                sh("rm -rf *")
            }
        }*/
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
                    sh ". activate ./.venv"
                    sh 'make test'
                }
            }
        }
    }
}