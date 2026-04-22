from django.shortcuts import render
from django.db.models import Sum, Count, Avg, Q
from rest_framework.decorators import api_view
from rest_framework.response import Response


# ── Dashboard HTML ────────────────────────────────────────────────────────────

def dashboard(request):
    return render(request, "dashboard/index.html")


# ── KPI Summary ───────────────────────────────────────────────────────────────

@api_view(["GET"])
def kpis(request):
    from apps.invoices.models import Invoice
    from apps.payments.models import Payment
    from apps.risk_scoring.models import RiskScore
    from apps.anomaly_detection.models import AnomalyScore
    from apps.customers.models import Customer
    from apps.refunds.models import Refund

    inv_agg = Invoice.objects.filter(is_deleted=False).aggregate(
        total_invoiced=Sum("total_amount"),
        total_collected=Sum("paid_amount"),
        total_outstanding=Sum("outstanding_amount"),
        total_discounted=Sum("discount_amount"),
        overdue_count=Count("id", filter=Q(status="overdue")),
    )

    high_risk = RiskScore.objects.filter(
        risk_severity__in=["high", "critical"],
        model_name="xgboost",
    ).count()

    anomaly_count = AnomalyScore.objects.filter(is_anomaly=True).count()

    total_customers = Customer.objects.filter(is_deleted=False).count()
    active_customers = Customer.objects.filter(is_deleted=False, status="active").count()
    churned_customers = Customer.objects.filter(is_deleted=False, status="churned").count()

    dup_refunds = Refund.objects.filter(is_duplicate=True).aggregate(
        count=Count("id"), amount=Sum("amount")
    )

    total_invoiced = float(inv_agg["total_invoiced"] or 0)
    total_collected = float(inv_agg["total_collected"] or 0)
    collection_rate = total_collected / total_invoiced if total_invoiced else 0

    return Response({
        "revenue": {
            "total_invoiced": round(total_invoiced, 2),
            "total_collected": round(total_collected, 2),
            "total_outstanding": round(float(inv_agg["total_outstanding"] or 0), 2),
            "total_discounted": round(float(inv_agg["total_discounted"] or 0), 2),
            "collection_rate_pct": round(collection_rate * 100, 2),
            "overdue_invoices": inv_agg["overdue_count"],
        },
        "risk": {
            "high_risk_invoices": high_risk,
            "anomalies_detected": anomaly_count,
            "duplicate_refunds": dup_refunds["count"] or 0,
            "duplicate_refund_amount": round(float(dup_refunds["amount"] or 0), 2),
        },
        "customers": {
            "total": total_customers,
            "active": active_customers,
            "churned": churned_customers,
            "churn_rate_pct": round(churned_customers / total_customers * 100, 2) if total_customers else 0,
        },
    })


# ── Revenue Trend (actual vs forecast) ───────────────────────────────────────

@api_view(["GET"])
def revenue_trend(request):
    from apps.forecasting.models import ForecastResult
    from django.db import connection

    # Actual monthly revenue from invoices
    with connection.cursor() as cur:
        cur.execute("""
            SELECT DATE_FORMAT(issue_date, '%Y-%m') AS month,
                   SUM(total_amount)                AS invoiced,
                   SUM(paid_amount)                 AS collected
            FROM invoices
            WHERE is_deleted = 0
            GROUP BY DATE_FORMAT(issue_date, '%Y-%m')
            ORDER BY month
        """)
        cols = [c[0] for c in cur.description]
        actuals = [dict(zip(cols, row)) for row in cur.fetchall()]

    # Forecast values
    forecasts = list(
        ForecastResult.objects
        .filter(metric="revenue", model_name="prophet")
        .order_by("period_month")
        .values("period_month", "forecasted_value", "lower_bound",
                "upper_bound", "is_anomalous", "deviation_pct")
    )

    return Response({"actuals": actuals, "forecasts": forecasts})


# ── Top High-Risk Invoices ────────────────────────────────────────────────────

@api_view(["GET"])
def top_risks(request):
    from apps.risk_scoring.models import RiskScore

    limit = int(request.query_params.get("limit", 15))
    scores = (
        RiskScore.objects
        .filter(model_name="xgboost")
        .select_related("customer", "invoice")
        .order_by("-leakage_probability")[:limit]
    )

    data = []
    for s in scores:
        data.append({
            "invoice_id": s.invoice_id,
            "invoice_number": s.invoice.invoice_number if s.invoice else None,
            "customer_name": s.customer.name,
            "customer_segment": s.customer.segment,
            "leakage_probability": round(float(s.leakage_probability), 4),
            "risk_severity": s.risk_severity,
            "rank_percentile": round(float(s.rank_percentile), 1) if s.rank_percentile else None,
            "top_features": s.top_features[:3] if s.top_features else [],
            "invoice_amount": float(s.invoice.total_amount) if s.invoice else None,
            "invoice_status": s.invoice.status if s.invoice else None,
        })
    return Response(data)


# ── Risk Severity Distribution ────────────────────────────────────────────────

@api_view(["GET"])
def risk_distribution(request):
    from apps.risk_scoring.models import RiskScore

    dist = (
        RiskScore.objects
        .filter(model_name="xgboost")
        .values("risk_severity")
        .annotate(count=Count("id"))
        .order_by("risk_severity")
    )
    return Response(list(dist))


# ── Leakage by Rule Code ──────────────────────────────────────────────────────

@api_view(["GET"])
def leakage_by_rule(request):
    from apps.leakage_cases.models import RuleAlert

    rules = (
        RuleAlert.objects
        .values("rule_code", "severity")
        .annotate(
            count=Count("id"),
            total_amount=Sum("leakage_amount"),
            unresolved=Count("id", filter=Q(is_resolved=False)),
        )
        .order_by("-total_amount")
    )
    return Response(list(rules))


# ── Invoice Detail (for modal) ────────────────────────────────────────────────

@api_view(["GET"])
def invoice_detail(request, invoice_id):
    from apps.invoices.models import Invoice
    from apps.risk_scoring.models import RiskScore
    from apps.anomaly_detection.models import AnomalyScore
    from apps.leakage_cases.models import RuleAlert

    try:
        invoice = Invoice.objects.select_related("customer", "subscription").get(id=invoice_id)
    except Invoice.DoesNotExist:
        return Response({"error": "Not found"}, status=404)

    risk = RiskScore.objects.filter(invoice_id=invoice_id, model_name="xgboost").first()
    anomaly = AnomalyScore.objects.filter(invoice_id=invoice_id).first()
    alerts = list(
        RuleAlert.objects.filter(invoice_id=invoice_id)
        .values("rule_code", "severity", "description", "leakage_amount", "is_resolved")
    )

    return Response({
        "invoice": {
            "id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "status": invoice.status,
            "issue_date": str(invoice.issue_date),
            "due_date": str(invoice.due_date),
            "total_amount": float(invoice.total_amount),
            "paid_amount": float(invoice.paid_amount),
            "outstanding_amount": float(invoice.outstanding_amount),
            "discount_amount": float(invoice.discount_amount),
        },
        "customer": {
            "id": invoice.customer_id,
            "name": invoice.customer.name,
            "segment": invoice.customer.segment,
            "status": invoice.customer.status,
        },
        "risk": {
            "leakage_probability": float(risk.leakage_probability) if risk else None,
            "risk_severity": risk.risk_severity if risk else None,
            "rank_percentile": float(risk.rank_percentile) if risk and risk.rank_percentile else None,
            "feature_snapshot": risk.feature_snapshot if risk else {},
            "top_features": risk.top_features if risk else [],
        } if risk else None,
        "anomaly": {
            "score": float(anomaly.score) if anomaly else None,
            "is_anomaly": bool(anomaly.is_anomaly) if anomaly else None,
            "if_prob_proxy": anomaly.feature_snapshot.get("if_prob_proxy") if anomaly else None,
            "top_features": anomaly.feature_snapshot.get("top_features", []) if anomaly else [],
        } if anomaly else None,
        "rule_alerts": alerts,
    })


# ── KPI Detail Endpoints ──────────────────────────────────────────────────────

@api_view(["GET"])
def kpi_revenue_breakdown(request):
    """Monthly invoiced/collected breakdown for revenue and collection-rate cards."""
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("""
            SELECT DATE_FORMAT(issue_date, '%Y-%m') AS month,
                   COUNT(*)                          AS invoice_count,
                   SUM(total_amount)                 AS invoiced,
                   SUM(paid_amount)                  AS collected,
                   SUM(outstanding_amount)            AS outstanding
            FROM invoices WHERE is_deleted=0
            GROUP BY DATE_FORMAT(issue_date, '%Y-%m')
            ORDER BY month
        """)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    for r in rows:
        inv = float(r["invoiced"] or 0)
        r["collection_rate"] = round(float(r["collected"] or 0) / inv * 100, 1) if inv else 0
        r["invoiced"] = float(r["invoiced"] or 0)
        r["collected"] = float(r["collected"] or 0)
        r["outstanding"] = float(r["outstanding"] or 0)
    return Response(rows)


@api_view(["GET"])
def kpi_outstanding(request):
    """Top outstanding (unpaid) invoices."""
    from apps.invoices.models import Invoice
    from django.utils import timezone
    limit = int(request.query_params.get("limit", 100))
    today = timezone.now().date()
    invoices = (
        Invoice.objects
        .filter(is_deleted=False, outstanding_amount__gt=0)
        .select_related("customer")
        .order_by("-outstanding_amount")[:limit]
    )
    data = []
    for inv in invoices:
        days = (today - inv.due_date).days if inv.due_date and inv.due_date < today else 0
        data.append({
            "invoice_number": inv.invoice_number,
            "invoice_id": inv.id,
            "customer_name": inv.customer.name,
            "customer_segment": inv.customer.segment,
            "total_amount": float(inv.total_amount),
            "outstanding_amount": float(inv.outstanding_amount),
            "due_date": str(inv.due_date),
            "status": inv.status,
            "days_overdue": max(0, days),
        })
    total_out = sum(r["outstanding_amount"] for r in data)
    return Response({"invoices": data, "total_outstanding": total_out, "count": len(data)})


@api_view(["GET"])
def kpi_overdue(request):
    """All overdue invoices with days overdue."""
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("""
            SELECT i.id, i.invoice_number, c.name AS customer_name, c.segment,
                   i.total_amount, i.outstanding_amount, i.due_date,
                   DATEDIFF(CURDATE(), i.due_date) AS days_overdue
            FROM invoices i
            JOIN customers c ON i.customer_id = c.id
            WHERE i.is_deleted=0 AND i.status='overdue'
            ORDER BY days_overdue DESC
            LIMIT 200
        """)
        cols = [c[0] for c in cur.description]
        rows = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            d["total_amount"] = float(d["total_amount"])
            d["outstanding_amount"] = float(d["outstanding_amount"])
            d["due_date"] = str(d["due_date"])
            rows.append(d)
    total = sum(r["outstanding_amount"] for r in rows)
    return Response({"invoices": rows, "total_overdue_amount": total, "count": len(rows)})


@api_view(["GET"])
def kpi_high_risk_detail(request):
    """High/critical risk invoice summary + top list."""
    from apps.risk_scoring.models import RiskScore
    limit = int(request.query_params.get("limit", 100))
    dist = list(
        RiskScore.objects.filter(model_name="xgboost")
        .values("risk_severity")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    scores = (
        RiskScore.objects
        .filter(model_name="xgboost", risk_severity__in=["critical", "high"])
        .select_related("customer", "invoice")
        .order_by("-leakage_probability")[:limit]
    )
    invoices = []
    for s in scores:
        invoices.append({
            "invoice_id": s.invoice_id,
            "invoice_number": s.invoice.invoice_number if s.invoice else None,
            "customer_name": s.customer.name,
            "customer_segment": s.customer.segment,
            "leakage_probability": round(float(s.leakage_probability), 4),
            "risk_severity": s.risk_severity,
            "rank_percentile": round(float(s.rank_percentile), 1) if s.rank_percentile else None,
            "top_features": (s.top_features or [])[:2],
            "invoice_amount": float(s.invoice.total_amount) if s.invoice else None,
        })
    return Response({"distribution": dist, "invoices": invoices})


@api_view(["GET"])
def kpi_anomalies_detail(request):
    """Isolation Forest anomaly details."""
    from apps.anomaly_detection.models import AnomalyScore
    limit = int(request.query_params.get("limit", 100))
    total = AnomalyScore.objects.filter(is_anomaly=True).count()
    total_scored = AnomalyScore.objects.count()
    qs = (
        AnomalyScore.objects
        .filter(is_anomaly=True)
        .select_related("customer", "invoice")
        .order_by("score")[:limit]
    )
    data = []
    for a in qs:
        data.append({
            "invoice_id": a.invoice_id,
            "invoice_number": a.invoice.invoice_number if a.invoice else None,
            "customer_name": a.customer.name if a.customer else None,
            "customer_segment": a.customer.segment if a.customer else None,
            "if_score": round(float(a.score), 4),
            "if_prob_proxy": round(float(a.feature_snapshot.get("if_prob_proxy", 0)), 4),
            "top_features": a.feature_snapshot.get("top_features", [])[:3],
            "invoice_amount": float(a.invoice.total_amount) if a.invoice else None,
        })
    return Response({
        "total_anomalies": total,
        "total_scored": total_scored,
        "anomaly_rate_pct": round(total / total_scored * 100, 1) if total_scored else 0,
        "anomalies": data,
    })


@api_view(["GET"])
def kpi_duplicate_refunds_detail(request):
    """All duplicate refunds."""
    from apps.refunds.models import Refund
    refunds = (
        Refund.objects
        .filter(is_duplicate=True)
        .select_related("customer", "invoice")
        .order_by("-amount")
    )
    data = []
    total_amount = 0
    for r in refunds:
        total_amount += float(r.amount)
        data.append({
            "refund_number": r.refund_number,
            "customer_name": r.customer.name,
            "customer_segment": r.customer.segment,
            "invoice_number": r.invoice.invoice_number if r.invoice else None,
            "amount": float(r.amount),
            "reason": r.reason,
            "status": r.status,
            "requested_at": str(r.requested_at)[:16] if r.requested_at else None,
            "processed_at": str(r.processed_at)[:16] if r.processed_at else None,
        })
    return Response({"refunds": data, "total_amount": total_amount, "count": len(data)})


# ── Anomaly Timeline ──────────────────────────────────────────────────────────

@api_view(["GET"])
def anomaly_timeline(request):
    from apps.anomaly_detection.models import AnomalyScore
    from django.db import connection

    with connection.cursor() as cur:
        cur.execute("""
            SELECT DATE_FORMAT(scored_at, '%Y-%m') AS month,
                   COUNT(*)                        AS total_scored,
                   SUM(is_anomaly)                 AS anomalies
            FROM anomaly_scores
            GROUP BY DATE_FORMAT(scored_at, '%Y-%m')
            ORDER BY month
        """)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    return Response(rows)
