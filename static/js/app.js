/* app.js - grafik Chart.js dan interaksi form */
(function () {
  "use strict";

  // ---- Tombol contoh cepat mengisi input ----
  document.querySelectorAll("[data-fill]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var input = document.getElementById("tv");
      if (!input) return;
      input.value = btn.getAttribute("data-fill");
      input.focus();
    });
  });

  // ---- Grafik ----
  var canvas = document.getElementById("salesChart");
  if (!canvas || typeof Chart === "undefined" || !window.APP) return;

  var data = window.APP;
  var train = data.points.filter(function (p) { return p.split === "train"; });
  var test = data.points.filter(function (p) { return p.split === "test"; });

  var datasets = [
    {
      label: "Data latih",
      data: train,
      backgroundColor: "rgba(99, 102, 241, 0.65)",
      pointRadius: 4,
      pointHoverRadius: 6,
      order: 3
    },
    {
      label: "Data uji",
      data: test,
      backgroundColor: "rgba(245, 158, 11, 0.9)",
      borderColor: "#ffffff",
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7,
      order: 2
    },
    {
      label: "Garis regresi",
      data: data.line,
      type: "scatter",
      showLine: true,
      borderColor: "#ef4444",
      borderWidth: 2.5,
      pointRadius: 0,
      pointHoverRadius: 0,
      fill: false,
      order: 4
    }
  ];

  if (data.prediction) {
    datasets.push({
      label: "Prediksi Anda",
      data: [data.prediction],
      backgroundColor: "#10b981",
      borderColor: "#064e3b",
      borderWidth: 2,
      pointStyle: "rectRot",
      pointRadius: 11,
      pointHoverRadius: 13,
      order: 1
    });
  }

  new Chart(canvas, {
    type: "scatter",
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: true },
      scales: {
        x: {
          title: { display: true, text: "Biaya Iklan TV (ribu USD)", font: { weight: "600" } },
          grid: { color: "rgba(148,163,184,.2)" },
          min: 0
        },
        y: {
          title: { display: true, text: "Penjualan (ribu unit)", font: { weight: "600" } },
          grid: { color: "rgba(148,163,184,.2)" }
        }
      },
      plugins: {
        legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8 } },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              return ctx.dataset.label + ": TV " + ctx.parsed.x.toFixed(1) +
                " \u2192 Sales " + ctx.parsed.y.toFixed(2);
            }
          }
        }
      }
    }
  });
})();
