@Library('xmos_jenkins_shared_library@v0.43.1') _

getApproval()

pipeline {
    agent {
        label "xcore.ai"
    }
    options {
        // skipDefaultCheckout()
        buildDiscarder(xmosDiscardBuildSettings(onlyArtifacts=false))
        timestamps()
    }
    environment {
        REPO = 'lib_tflite_micro'
        VIEW = getViewName(REPO)
    }
    stages {
            stage('Build') {
                steps {
                    createVenv(reqFile: "requirements.txt")
                    withVenv {
                        sh 'git submodule update --depth=1 --init --recursive --jobs 8'
                        sh 'make init'
                        sh 'make patch'
                        sh 'make build'
                    }
                }
            }
            stage("Test") {
                steps {
                    withVenv {
                        sh 'make init'
                        catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
                            sh 'make test'
                        }
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
