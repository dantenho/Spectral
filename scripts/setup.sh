#!/bin/bash

# ============================================================================
# Script de Setup Automatizado - Spectral Server
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "  🚀 SPECTRAL SERVER - SETUP AUTOMATIZADO"
echo "============================================================================"
echo ""

# ============================================================================
# 1. VERIFICAR PYTHON
# ============================================================================
echo "📋 Verificando Python..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Por favor instale Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✅ Python encontrado: $PYTHON_VERSION"

# ============================================================================
# 2. CRIAR AMBIENTE VIRTUAL
# ============================================================================
echo ""
echo "🐍 Criando ambiente virtual..."

cd "$(dirname "$0")/.."

if [ -d "server/venv" ]; then
    echo "⚠️  Ambiente virtual já existe. Removendo..."
    rm -rf server/venv
fi

cd server
python3 -m venv venv

echo "✅ Ambiente virtual criado"

# ============================================================================
# 3. ATIVAR E INSTALAR DEPENDÊNCIAS
# ============================================================================
echo ""
echo "📦 Instalando dependências..."

source venv/bin/activate

# Atualizar pip
pip install --upgrade pip setuptools wheel

# Instalar dependências
pip install -r requirements.txt

echo "✅ Dependências instaladas"

# ============================================================================
# 4. CONFIGURAR VARIÁVEIS DE AMBIENTE
# ============================================================================
echo ""
echo "⚙️  Configurando variáveis de ambiente..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Arquivo .env criado a partir do exemplo"
    echo "⚠️  IMPORTANTE: Edite o arquivo .env com suas configurações!"
else
    echo "ℹ️  Arquivo .env já existe"
fi

# ============================================================================
# 5. CRIAR DIRETÓRIOS
# ============================================================================
echo ""
echo "📁 Criando estrutura de diretórios..."

cd ..

mkdir -p data/events
mkdir -p data/training/{train,val,test}
mkdir -p models/{checkpoints,pretrained,production}
mkdir -p logs

echo "✅ Diretórios criados"

# ============================================================================
# 6. VERIFICAR DEPENDÊNCIAS OPCIONAIS
# ============================================================================
echo ""
echo "🔍 Verificando dependências opcionais..."

# InfluxDB
if command -v influx &> /dev/null; then
    echo "✅ InfluxDB CLI encontrado"
else
    echo "⚠️  InfluxDB CLI não encontrado (opcional)"
fi

# PostgreSQL
if command -v psql &> /dev/null; then
    echo "✅ PostgreSQL CLI encontrado"
else
    echo "⚠️  PostgreSQL CLI não encontrado (opcional)"
fi

# Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker encontrado"

    echo ""
    echo "🐳 Deseja iniciar serviços com Docker? (y/n)"
    read -p "Resposta: " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Iniciando containers Docker..."
        docker-compose up -d influxdb postgres

        echo "⏳ Aguardando serviços ficarem prontos (10s)..."
        sleep 10

        echo "✅ Serviços Docker iniciados"
    fi
else
    echo "⚠️  Docker não encontrado (opcional)"
fi

# ============================================================================
# 7. VALIDAR SETUP
# ============================================================================
echo ""
echo "🧪 Validando setup..."

cd server
source venv/bin/activate

python config/settings.py

echo "✅ Validação completa"

# ============================================================================
# CONCLUSÃO
# ============================================================================
echo ""
echo "============================================================================"
echo "  ✅ SETUP COMPLETO!"
echo "============================================================================"
echo ""
echo "📝 Próximos passos:"
echo ""
echo "1. Edite o arquivo server/.env com suas configurações"
echo ""
echo "2. Inicie o servidor:"
echo "   cd server"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "3. Ou use o script de inicialização:"
echo "   ./scripts/start_server.sh"
echo ""
echo "4. Acesse a documentação da API:"
echo "   http://localhost:8000/docs"
echo ""
echo "============================================================================"
