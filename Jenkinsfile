@Library('xmos_jenkins_shared_library@v0.43.1') _

getApproval()

def withNativeTools(platform, Closure body) {
    if (platform == 'windows') 
    {
        withVS() {body()}
    } 
    else {
        body()
    }
}

def buildNative(platform) {
    stage('build_native') {
        withNativeTools(platform) {
            sh 'make build'
        }
    }
}

def testNative(platform) {
    stage('Test Host Interp.') {
        withNativeTools(platform) {
            sh 'make test'
        }
    }
    stage('Tests Native') {
        withNativeTools(platform) {
            dir("tests"){
                withVenv {
                    sh "pytest -s --hw-target=native"
                }
            }
        }
    }
}

def buildXcore(makeTarget, toolsVersion) {
    stage(makeTarget) {
        withTools(toolsVersion) {
            sh "make ${makeTarget}"
        }
    }
}

def testXcore(hwTarget, toolsVersion) {
    stage("Test ${hwTarget}") {
        withTools(toolsVersion) {
            dir("tests"){
                withVenv {
                    sh "pytest -s --hw-target=${hwTarget}"
                }
            }
        }
    }
}

def buildInstall(toolsVersion) {
    stage('Build Install') {
        withTools(toolsVersion) {
            sh 'make build_install'
        }
    }
}

def buildAndTest(platform) {
    try {
        dir(env.REPO) {
            stage('Setup') {
                checkoutScmShallow()
                sh 'git submodule update --depth=1 --init --recursive --jobs 8'
                dir('tests') {
                    createVenv(reqFile: 'requirements.txt')
                }
            }

            // Native
            buildNative(platform)
            testNative(platform)

            // XS3
            buildXcore('build_xs3', params.TOOLS_VERSION_XS)
            testXcore('xs3', params.TOOLS_VERSION_XS)
            // uninstallTools(params.TOOLS_VERSION_XS)

            // VX4
            buildXcore('build_vx4', params.TOOLS_VERSION_VX)
            testXcore('vx4-xcc', params.TOOLS_VERSION_VX)
            // uninstallTools(params.TOOLS_VERSION_VX)

            // Install
            buildInstall(params.TOOLS_VERSION_XS)

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
            defaultValue: 'slipgate@develop',
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
