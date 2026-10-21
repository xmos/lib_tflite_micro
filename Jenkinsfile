@Library('xmos_jenkins_shared_library@v0.43.1') _

getApproval()

def withNativeTools(platform, Closure body) {
    if (platform == 'windows') {
        withVS() {body()}
    } 
    else {
        body()
    }
}

def runtimeTests(target){
    dir("tests"){
        createVenv()
        withVenv {
            sh "pip install pytest"
            sh "pytest -s --hw-target=${target}"
        }
    }
}

def buildAndTest(platform) {
    try {
        dir(env.REPO) {
            stage('Setup') {
                checkoutScmShallow()
                sh 'git submodule update --depth=1 --init --recursive --jobs 8'
            }

            stage('Build Native') {
                withNativeTools(platform) {
                    sh 'make build'
                }
            }
            
            stage('Build VX4') {
                withTools(params.TOOLS_VERSION_VX) {
                    sh 'make build_vx4'
                }
            }

            stage('Build XS3') {
                withTools(params.TOOLS_VERSION_XS) {
                    sh 'make build_xs3'
                }
            }

            stage('Build Install') {
                withTools(params.TOOLS_VERSION_XS) {
                    sh 'make build_install'
                }
            }

            // ------------ TESTS ---------------

            stage('Host Interp. Test') {
                withNativeTools(platform) {
                    sh 'make test'
                }
            }
            
            stage('Runtime Tests - Native') {
                withNativeTools(platform) {
                    runtimeTests('native')
                }
            }

            stage('Runtime Tests - VX4') {
                withTools(params.TOOLS_VERSION_VX) {
                    runtimeTests('vx4-xcc')
                }
            }

            stage('Runtime Tests - XS3') {
                withTools(params.TOOLS_VERSION_XS) {
                    runtimeTests('xs3')
                }
            }

            // archive
            def archivePath = "build_xs3/release_archive_${platform}.zip"
            sh "cmake -E rename build_xs3/release_archive.zip ${archivePath}"
            archiveArtifacts artifacts: archivePath, fingerprint: true
        }
    } 
    finally {
        xcoreCleanSandbox()
    }
}

pipeline {
    agent none

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
            defaultValue: '-j --repo arch_vx_slipgate -b master -a XTC 135',
            description: 'VX XTC tools version'
        )
    }

    options {
        timestamps()
        skipDefaultCheckout()
        buildDiscarder(xmosDiscardBuildSettings())
    }

    stages {
        stage('Build and test') {
            steps {
                script {
                    parallel(
                        Linux:      { node('linux && x86_64')       { buildAndTest('linux') } },
                        Mac:        { node('macos && arm64')        { buildAndTest('mac') } },
                        Windows:    { node('windows10 && unified')  { buildAndTest('windows') } }
                    )
                }
            }
        }
    }
}
