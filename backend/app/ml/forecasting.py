def forecast_status():
    return {"available": False, "reason": "The IBM Telco dataset is a point-in-time snapshot without dated monthly observations; forecasting is withheld until historical snapshots are supplied."}
