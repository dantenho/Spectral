#!/bin/bash

# ============================================================================
# Script de Inicialização - Spectral Server
# ============================================================================

echo "============================================================================"
echo "  🚀 INICIANDO SPECTRAL SERVER"
echo "============================================================================"
echo ""

# Ir para diretório do servidor
cd "$(dirname "$0")/../server"

# Verificar se ambiente virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Ambiente virtual não encontrado!"
    echo "   Execute primeiro: ./scripts/setup.sh"
    exit 1
fi

# Ativar ambiente virtual
echo "🐍 Ativando ambiente virtual..."
source venv/bin/activate

# Verificar .env
if [ ! -f ".env" ]; then
    echo "⚠️  Arquivo .env não encontrado. Usando .env.example..."
    cp .env.example .env
fi

# Verificar dependências
echo "📦 Verificando dependências..."
python -c "import fastapi, uvicorn, numpy, scipy, librosa" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "❌ Algumas dependências estão faltando!"
    echo "   Instalando..."
    pip install -r requirements.txt
fi

echo "✅ Dependências OK"
echo ""

# Validar configurações
echo "⚙️  Validando configurações..."
python config/settings.py
echo ""

# Iniciar servidor
echo "============================================================================"
echo "  🚀 SERVIDOR INICIADO"
echo "============================================================================"
echo ""
echo "  🌐 API: http://localhost:8000"
echo "  📚 Docs: http://localhost:8000/docs"
echo "  📊 Stats: http://localhost:8000/stats"
echo "  🔌 WebSocket: ws://localhost:8000/ws/{client_id}"
echo ""
echo "  Pressione CTRL+C para parar"
echo ""
echo "============================================================================"
echo ""

# Executar servidor
python main.py
