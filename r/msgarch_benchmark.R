# Benchmark econométrico MS-GARCH (Haas, Mittnik e Paolella, 2004)
# Implementação: pacote MSGARCH (Ardia et al., 2019)
#
# Entrada: CSV com retornos diários (data/processed/returns.csv)
# Saída: VaR/ES condicionais ao regime via FHS (data/processed/msgarch_var_es.csv)

# install.packages("MSGARCH")
library(MSGARCH)

returns_path <- "../data/processed/returns.csv"
output_path <- "../data/processed/msgarch_var_es.csv"

confidence_levels <- c(0.95, 0.99)
n_states_grid <- c(2, 3)

# TODO (Semana 8):
# 1. Carregar retornos (por ativo e/ou agregado da carteira)
# 2. Especificar CreateSpec(model = "sGARCH", distribution.spec = "std", k = n_states)
# 3. Estimar via FitML para cada especificação de n_states_grid; selecionar por BIC
# 4. Calcular VaR/ES via método FHS (Risk() do pacote MSGARCH)
# 5. Exportar para output_path para comparação com o benchmark no pipeline Python
