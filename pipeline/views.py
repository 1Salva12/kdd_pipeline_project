from django.shortcuts import render
from .utils import process_kdd_data

def dashboard_view(request):
    # Llamamos a la función que procesa el dataset (Fase 1)
    try:
        data_results = process_kdd_data()
        context = {
            'results': data_results,
            'status': 'success'
        }
    except Exception as e:
        context = {
            'error_message': str(e),
            'status': 'error'
        }
    
    return render(request, 'pipeline/dashboard.html', context)