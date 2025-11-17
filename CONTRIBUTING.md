# Contribuindo para o Spectral

Obrigado por considerar contribuir para o projeto Spectral! Este documento fornece diretrizes para contribuições.

## 🤝 Como Contribuir

### 1. Fork e Clone

```bash
# Fork o repositório no GitHub
# Depois clone seu fork
git clone https://github.com/seu-usuario/Spectral.git
cd Spectral
```

### 2. Criar Branch

```bash
# Crie uma branch para sua feature/fix
git checkout -b feature/nome-da-feature
# ou
git checkout -b fix/nome-do-fix
```

### 3. Desenvolver

- Escreva código limpo e bem documentado
- Siga os padrões de estilo do projeto
- Adicione testes quando apropriado
- Atualize documentação se necessário

### 4. Commit

```bash
# Adicione suas mudanças
git add .

# Commit com mensagem descritiva
git commit -m "feat: adiciona detecção de anomalia em infrasom"
```

#### Convenção de Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - Nova funcionalidade
- `fix:` - Correção de bug
- `docs:` - Mudanças em documentação
- `style:` - Formatação, sem mudanças de código
- `refactor:` - Refatoração de código
- `test:` - Adicionar ou corrigir testes
- `chore:` - Tarefas de manutenção

### 5. Push e Pull Request

```bash
# Push para seu fork
git push origin feature/nome-da-feature

# Crie Pull Request no GitHub
```

## 📋 Checklist de Pull Request

- [ ] Código segue os padrões do projeto
- [ ] Todos os testes passam
- [ ] Documentação atualizada
- [ ] Commits seguem convenção
- [ ] Branch está atualizado com main
- [ ] Descrição clara do PR

## 🎨 Padrões de Código

### Python

- Siga [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints
- Docstrings em funções públicas
- Máximo 100 caracteres por linha

```python
def process_sensor_data(
    data: SensorPacket,
    threshold: float = 3.0
) -> AnomalyResult:
    """
    Processa dados de sensores e detecta anomalias.

    Args:
        data: Pacote de dados dos sensores
        threshold: Limiar de detecção em desvios padrão

    Returns:
        Resultado da análise de anomalia
    """
    pass
```

### Kotlin

- Siga [Kotlin Style Guide](https://kotlinlang.org/docs/coding-conventions.html)
- Use val ao invés de var quando possível
- Prefira expressões a statements
- Use coroutines para operações assíncronas

```kotlin
class SensorDataCollector(
    private val context: Context
) {
    suspend fun collectData(): SensorPacket {
        // Implementation
    }
}
```

## 🧪 Testes

### Python

```bash
# Executar testes
pytest

# Com cobertura
pytest --cov=server --cov-report=html
```

### Android

```bash
# Testes unitários
./gradlew test

# Testes instrumentados
./gradlew connectedAndroidTest
```

## 📝 Documentação

- Documente código público
- Atualize README.md se necessário
- Adicione exemplos de uso
- Mantenha docs/ atualizado

## 🐛 Reportando Bugs

Ao reportar bugs, inclua:

1. **Descrição**: O que aconteceu?
2. **Reprodução**: Passos para reproduzir
3. **Esperado**: Comportamento esperado
4. **Ambiente**: OS, versão, hardware
5. **Logs**: Mensagens de erro relevantes

## 💡 Sugerindo Features

Para sugerir features:

1. Verifique issues existentes
2. Crie issue detalhado
3. Explique o caso de uso
4. Proponha solução (opcional)

## 🔍 Code Review

Todas as contribuições passam por code review:

- Seja respeitoso e construtivo
- Responda feedback prontamente
- Discuta decisões de design
- Aprenda e ensine

## 📜 Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a licença MIT.

## 🙏 Agradecimentos

Todo tipo de contribuição é valorizado:

- 🐛 Reportar bugs
- 💡 Sugerir features
- 📝 Melhorar documentação
- 💻 Escrever código
- 🧪 Adicionar testes
- 🎨 Melhorar UX/UI

---

**Obrigado por contribuir para o Spectral!**
