"use client";

import { useEffect, useRef } from "react";
import type { StressCurve } from "../lib/analysis";

interface StressChartProps {
  curves: StressCurve[];
}

export function StressChart({ curves }: StressChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const container = canvas.parentElement;
    if (!container) return;

    const draw = () => {
      const bounds = container.getBoundingClientRect();
      const width = Math.max(360, bounds.width);
      const height = width < 620 ? 310 : 360;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.scale(ratio, ratio);
      context.clearRect(0, 0, width, height);

      const margin = { top: 24, right: 18, bottom: 48, left: 54 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const yMinimum = Math.min(-0.2, ...curves.flatMap((curve) => curve.points.map((point) => point.low - 0.05)));
      const yMaximum = 1.12;
      const xScale = (value: number) => margin.left + value * plotWidth;
      const yScale = (value: number) => margin.top + ((yMaximum - value) / (yMaximum - yMinimum)) * plotHeight;

      context.fillStyle = "#fbfaf5";
      context.fillRect(0, 0, width, height);
      context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      context.textBaseline = "middle";
      context.strokeStyle = "#dedbd0";
      context.fillStyle = "#6f716b";
      context.lineWidth = 1;

      for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
        const x = xScale(tick);
        context.beginPath();
        context.moveTo(x, margin.top);
        context.lineTo(x, margin.top + plotHeight);
        context.stroke();
        context.textAlign = "center";
        context.fillText(`${Math.round(tick * 100)}%`, x, height - 24);
      }

      for (const tick of [0, 0.5, 1]) {
        const y = yScale(tick);
        context.beginPath();
        context.moveTo(margin.left, y);
        context.lineTo(margin.left + plotWidth, y);
        context.stroke();
        context.textAlign = "right";
        context.fillText(tick.toFixed(1), margin.left - 10, y);
      }

      context.save();
      context.strokeStyle = "#8c8d87";
      context.setLineDash([3, 4]);
      context.beginPath();
      context.moveTo(margin.left, yScale(1));
      context.lineTo(margin.left + plotWidth, yScale(1));
      context.stroke();
      context.restore();

      for (const curve of curves) {
        const maximum = Math.max(...curve.points.map((point) => point.level), 1e-9);
        context.save();
        context.globalAlpha = 0.1;
        context.fillStyle = curve.stroke;
        context.beginPath();
        curve.points.forEach((point, index) => {
          const x = xScale(point.level / maximum);
          const y = yScale(point.high);
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        [...curve.points].reverse().forEach((point) => {
          context.lineTo(xScale(point.level / maximum), yScale(point.low));
        });
        context.closePath();
        context.fill();
        context.restore();

        context.save();
        context.strokeStyle = curve.stroke;
        context.fillStyle = "#fbfaf5";
        context.lineWidth = 2.4;
        context.setLineDash(curve.dash);
        context.lineJoin = "round";
        context.beginPath();
        curve.points.forEach((point, index) => {
          const x = xScale(point.level / maximum);
          const y = yScale(point.median);
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.stroke();
        context.setLineDash([]);
        for (const point of curve.points) {
          context.beginPath();
          context.arc(xScale(point.level / maximum), yScale(point.median), 3.3, 0, Math.PI * 2);
          context.fill();
          context.stroke();
        }
        context.restore();
      }

      context.fillStyle = "#6f716b";
      context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      context.textAlign = "center";
      context.fillText("NORMALIZED STRESS SEVERITY", margin.left + plotWidth / 2, height - 7);
      context.save();
      context.translate(13, margin.top + plotHeight / 2);
      context.rotate(-Math.PI / 2);
      context.fillText("RETAINED SKILL", 0, 0);
      context.restore();
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(container);
    return () => observer.disconnect();
  }, [curves]);

  return (
    <div className="chart-shell">
      <canvas
        ref={canvasRef}
        aria-label="Stress-response curves showing retained predictive skill across normalized stress severity"
        role="img"
      />
      <div className="chart-legend" aria-label="Chart legend">
        {curves.map((curve) => (
          <div className="legend-item" key={curve.id}>
            <span className="legend-line" style={{ backgroundColor: curve.stroke }} />
            <span>{curve.shortLabel}</span>
          </div>
        ))}
      </div>
      <p className="chart-note">
        Lines show split medians; translucent bands span the empirical 5th–95th percentiles. Severity is normalized within each operator.
      </p>
    </div>
  );
}
