package com.spectral.app.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.spectral.app.R
import com.spectral.app.service.SensorService
import com.spectral.app.util.PermissionsHelper

/**
 * MainActivity - Tela principal do Spectral
 *
 * Features:
 * - Verificação de permissões
 * - Calibração de sensores
 * - Início de detecção
 * - Configurações
 * - Histórico
 */
class MainActivity : AppCompatActivity() {

    private val PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.BLUETOOTH,
        Manifest.permission.BLUETOOTH_CONNECT,
        Manifest.permission.NFC
    )

    private val PERMISSION_REQUEST_CODE = 1001

    // Views
    private lateinit var btnStartDetection: MaterialButton
    private lateinit var btnCalibrate: MaterialButton
    private lateinit var btnSettings: MaterialButton
    private lateinit var btnHistory: MaterialButton
    private lateinit var cardStatus: MaterialCardView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Setup theme escuro
        supportActionBar?.hide()

        // Inicializar views
        initViews()

        // Verificar permissões
        checkPermissions()
    }

    private fun initViews() {
        btnStartDetection = findViewById(R.id.btn_start_detection)
        btnCalibrate = findViewById(R.id.btn_calibrate)
        btnSettings = findViewById(R.id.btn_settings)
        btnHistory = findViewById(R.id.btn_history)
        cardStatus = findViewById(R.id.card_status)

        // Listeners
        btnStartDetection.setOnClickListener {
            if (checkSensorsCalibratedexecute() {
                    startDetectionActivity()
                }
            } else {
                showCalibrationRequiredDialog()
            }
        }

        btnCalibrate.setOnClickListener {
            startCalibrationActivity()
        }

        btnSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        btnHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }
    }

    private fun checkPermissions() {
        val missingPermissions = PERMISSIONS.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                missingPermissions.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        } else {
            onPermissionsGranted()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == PERMISSION_REQUEST_CODE) {
            val allGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }

            if (allGranted) {
                onPermissionsGranted()
            } else {
                showPermissionsDeniedDialog()
            }
        }
    }

    private fun onPermissionsGranted() {
        // Permissões OK, habilitar funcionalidades
        btnStartDetection.isEnabled = true
        btnCalibrate.isEnabled = true
    }

    private fun checkSensorsCalibrated(): Boolean {
        val prefs = getSharedPreferences("spectral_prefs", MODE_PRIVATE)
        return prefs.getBoolean("sensors_calibrated", false)
    }

    private fun startDetectionActivity() {
        val intent = Intent(this, DetectionActivity::class.java)
        startActivity(intent)
    }

    private fun startCalibrationActivity() {
        val intent = Intent(this, CalibrationActivity::class.java)
        startActivityForResult(intent, CALIBRATION_REQUEST_CODE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == CALIBRATION_REQUEST_CODE && resultCode == RESULT_OK) {
            // Calibração completa
            showCalibrationCompleteDialog()
        }
    }

    private fun showPermissionsDeniedDialog() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Permissões Necessárias")
            .setMessage("O Spectral precisa de permissões para acessar sensores, câmera e áudio.")
            .setPositiveButton("Tentar Novamente") { _, _ ->
                checkPermissions()
            }
            .setNegativeButton("Sair") { _, _ ->
                finish()
            }
            .setCancelable(false)
            .show()
    }

    private fun showCalibrationRequiredDialog() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Calibração Necessária")
            .setMessage("Os sensores precisam ser calibrados antes de iniciar a detecção.")
            .setPositiveButton("Calibrar Agora") { _, _ ->
                startCalibrationActivity()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

    private fun showCalibrationCompleteDialog() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Calibração Completa")
            .setMessage("Os sensores foram calibrados com sucesso!")
            .setPositiveButton("OK", null)
            .show()
    }

    companion object {
        private const val CALIBRATION_REQUEST_CODE = 2001
    }
}
