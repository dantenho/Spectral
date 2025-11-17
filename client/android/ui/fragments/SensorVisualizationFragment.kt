package com.spectral.app.ui.fragments

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.spectral.app.R
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Fragment de visualização de sensores em tempo real
 *
 * Mostra gráficos de linha para:
 * - Acelerômetro (X, Y, Z)
 * - Giroscópio (X, Y, Z)
 * - Magnetômetro (X, Y, Z)
 */
class SensorVisualizationFragment : Fragment() {

    // Charts
    private lateinit var chartAccel: LineChart
    private lateinit var chartGyro: LineChart
    private lateinit var chartMag: LineChart

    // Data buffers
    private val accelDataX = ConcurrentLinkedQueue<Entry>()
    private val accelDataY = ConcurrentLinkedQueue<Entry>()
    private val accelDataZ = ConcurrentLinkedQueue<Entry>()

    private val gyroDataX = ConcurrentLinkedQueue<Entry>()
    private val gyroDataY = ConcurrentLinkedQueue<Entry>()
    private val gyroDataZ = ConcurrentLinkedQueue<Entry>()

    private val magDataX = ConcurrentLinkedQueue<Entry>()
    private val magDataY = ConcurrentLinkedQueue<Entry>()
    private val magDataZ = ConcurrentLinkedQueue<Entry>()

    private var dataIndex = 0f
    private val MAX_DATA_POINTS = 100

    // Broadcast receiver
    private val sensorReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            intent ?: return

            val sensorType = intent.getStringExtra("sensor_type") ?: return
            val values = intent.getFloatArrayExtra("values") ?: return

            updateChart(sensorType, values)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_sensor_visualization, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Inicializar charts
        chartAccel = view.findViewById(R.id.chart_accel)
        chartGyro = view.findViewById(R.id.chart_gyro)
        chartMag = view.findViewById(R.id.chart_mag)

        setupChart(chartAccel, "Acelerômetro")
        setupChart(chartGyro, "Giroscópio")
        setupChart(chartMag, "Magnetômetro")

        // Registrar broadcast receiver
        LocalBroadcastManager.getInstance(requireContext())
            .registerReceiver(sensorReceiver, IntentFilter("SENSOR_DATA"))
    }

    private fun setupChart(chart: LineChart, title: String) {
        chart.apply {
            description.text = title
            description.textSize = 12f
            description.textColor = Color.WHITE

            setBackgroundColor(Color.BLACK)
            setGridBackgroundColor(Color.DKGRAY)

            axisLeft.textColor = Color.WHITE
            axisRight.textColor = Color.WHITE
            xAxis.textColor = Color.WHITE

            legend.textColor = Color.WHITE
            legend.isEnabled = true

            setTouchEnabled(false)
            isDragEnabled = false
            setScaleEnabled(false)

            // Limites do eixo Y
            axisLeft.axisMinimum = -20f
            axisLeft.axisMaximum = 20f
        }
    }

    private fun updateChart(sensorType: String, values: FloatArray) {
        dataIndex++

        when (sensorType) {
            "accelerometer" -> {
                addDataPoint(accelDataX, values[0])
                addDataPoint(accelDataY, values[1])
                addDataPoint(accelDataZ, values[2])
                updateChartData(chartAccel, accelDataX, accelDataY, accelDataZ)
            }
            "gyroscope" -> {
                addDataPoint(gyroDataX, values[0])
                addDataPoint(gyroDataY, values[1])
                addDataPoint(gyroDataZ, values[2])
                updateChartData(chartGyro, gyroDataX, gyroDataY, gyroDataZ)
            }
            "magnetometer" -> {
                addDataPoint(magDataX, values[0])
                addDataPoint(magDataY, values[1])
                addDataPoint(magDataZ, values[2])
                updateChartData(chartMag, magDataX, magDataY, magDataZ)
            }
        }
    }

    private fun addDataPoint(queue: ConcurrentLinkedQueue<Entry>, value: Float) {
        queue.add(Entry(dataIndex, value))

        // Limitar tamanho
        while (queue.size > MAX_DATA_POINTS) {
            queue.poll()
        }
    }

    private fun updateChartData(
        chart: LineChart,
        dataX: ConcurrentLinkedQueue<Entry>,
        dataY: ConcurrentLinkedQueue<Entry>,
        dataZ: ConcurrentLinkedQueue<Entry>
    ) {
        val dataSetX = LineDataSet(ArrayList(dataX), "X").apply {
            color = Color.RED
            setDrawCircles(false)
            setDrawValues(false)
            lineWidth = 2f
        }

        val dataSetY = LineDataSet(ArrayList(dataY), "Y").apply {
            color = Color.GREEN
            setDrawCircles(false)
            setDrawValues(false)
            lineWidth = 2f
        }

        val dataSetZ = LineDataSet(ArrayList(dataZ), "Z").apply {
            color = Color.BLUE
            setDrawCircles(false)
            setDrawValues(false)
            lineWidth = 2f
        }

        chart.data = LineData(dataSetX, dataSetY, dataSetZ)
        chart.notifyDataSetChanged()
        chart.invalidate()
    }

    override fun onDestroyView() {
        super.onDestroyView()

        // Unregister receiver
        LocalBroadcastManager.getInstance(requireContext())
            .unregisterReceiver(sensorReceiver)
    }
}
