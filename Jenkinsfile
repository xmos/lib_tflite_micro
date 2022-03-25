def date_str_bld = new Date().format("yyyy-mm-dd")

getApproval()

pipeline {
    agent {
        label "xcore.ai"
    }
    options {
        disableConcurrentBuilds()
        skipDefaultCheckout()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }
    stages {
        stage("Checkout repo") {
            steps {
                dir('lib_tflite_micro') {
                    checkout scm
                    stash includes: '**/*', name: 'lib_tflite_micro', useDefaultExcludes: false
                    script {
                        def short_hash = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
                        currentBuild.displayName = '#' + BUILD_NUMBER + '-' + short_hash
                    }
                }                        
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
