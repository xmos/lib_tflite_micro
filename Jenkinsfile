@Library('xmos_jenkins_shared_library@v0.43.1') _

getApproval()

pipeline {
    agent {
        label 'linux && x86_64'
    }

    environment {
        REPO = 'lib_tflite_micro'
    }

    parameters {
        string(
            name: 'TOOLS_VERSION_XS',
            defaultValue: '15.3.1',
            description: 'XS XTC tools version'
        )
        string(
            name: 'TOOLS_VERSION_VX',
            defaultValue: '-j --repo arch_vx_slipgate -b master -a XTC 131',
            description: 'VX XTC tools version'
        )
    }

    options {
        timestamps()
        skipDefaultCheckout()
        buildDiscarder(xmosDiscardBuildSettings())
    }

    stages {
        stage('Setup') {
            steps {
                dir(REPO) {
                    checkoutScmShallow()
                    createVenv(reqFile: 'requirements.txt')
                    sh 'git submodule update --depth=1 --init --recursive --jobs 8'
                    sh 'make patch'
                }
            }
        }

        stage('Build Native') {
            steps {
                dir(REPO) {
                    withVenv {
                        sh 'make build'
                    }
                }
            }
        }

        stage('Build XS3') {
            steps {
                dir(REPO) {
                    withTools(params.TOOLS_VERSION_XS) {
                        sh 'make build_xs3'
                    }
                }
            }
        }

        stage('Build VX4') {
            steps {
                dir(REPO) {
                    withTools(params.TOOLS_VERSION_VX) {
                        sh 'make build_vx4'
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
