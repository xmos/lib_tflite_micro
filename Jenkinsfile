@Library('xmos_jenkins_shared_library@v0.18.0') _

getApproval()

pipeline {
    agent {
        label "xcore.ai"
    }
    options {

        // skipDefaultCheckout()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }
    environment {
        REPO = 'lib_tflite_micro'
        VIEW = getViewName(REPO)
    }
    stages {
            stage('Build') {
                steps {
                    installPipfile(false)
                    withVenv {
                        sh 'git submodule update --depth=1 --init --recursive --jobs 8'
                        sh 'make init'
			sh 'make build'
                    }
                }
            }
            stage("Test") {
                steps {
                    withVenv {
			sh 'make init'
			sh 'make test'
                    }
                }
            }
    }
    post {
        cleanup {
            cleanWs()
        }
    }
}
        // stage("Checkout repo") {
        //     steps {
        //         dir('lib_tflite_micro') {
        //             checkout scm
        //             stash includes: '**/*', name: 'lib_tflite_micro', useDefaultExcludes: false
        //             script {
        //                 def short_hash = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
        //                 currentBuild.displayName = '#' + BUILD_NUMBER + '-' + short_hash
        //             }
        //         }                        
        //     }
        //     post {
        //         cleanup {
        //             deleteDir()
        //         }
        //     }
        // }
/*        stage("Cleanup2") {
            steps {
                // The Jenkins command deleteDir() doesn't seem very reliable, so we're using the basic form
//                sh("rm -rf *")
            }
        }*/
