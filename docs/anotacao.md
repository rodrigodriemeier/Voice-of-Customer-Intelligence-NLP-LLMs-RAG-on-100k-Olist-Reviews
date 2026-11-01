# Guia de anotação do conjunto de referência

Critério usado nas 400 avaliações de `data/annotations/gold_400.csv`. Novas
anotações devem seguir o mesmo critério para que as métricas continuem
comparáveis.

## Processo

1. **Primeira passada** assistida por LLM, considerando só o texto, **sem a
   nota em estrelas**, para que o rótulo de sentimento não seja contaminado
   pela nota. Essa passada está congelada em `gold_400_primeira_passada.csv`
   e não deve ser editada.
2. **Revisão humana** na página "Revisar anotações" do app:
   - **Modo cego** (padrão): os rótulos da primeira passada ficam ocultos e a
     avaliação é rotulada do zero. Com os rótulos visíveis, a revisão tende a
     confirmá-los e a concordância medida deixa de ser informativa.
   - **Ordem aleatória fixa**: revisando em sequência, as primeiras N formam
     uma amostra aleatória do gold. Com 100 revisadas já é possível estimar a
     concordância.
   - A coluna `reviewed` marca o que foi revisado e `review_mode` indica como
     (`cega` ou `com_rotulos`).
3. **Medição**: `python run_pipeline.py --from gold_review` compara a revisão
   com a primeira passada (concordância e kappa por campo), recalcula as
   métricas dos modelos no gold atualizado e só nas linhas revisadas, e
   atualiza o README. Se o LLM já tiver sido executado,
   `python run_pipeline.py --llm` faz o mesmo para ele, usando o cache.

Se o LLM avaliado na etapa 2B for da mesma família do que auxiliou a primeira
passada, ele pode concordar com o gold por compartilhar vieses, e não por estar
certo. As métricas calculadas só nas linhas revisadas em modo cego evitam esse
problema.

## Campos

**sentiment**: `negative`, `neutral` ou `positive`, pelo **texto**.

- `positive`: cliente satisfeito no geral, mesmo com uma ressalva pequena.
- `negative`: a reclamação domina. "Ainda não recebi" é negativo, mesmo
  sem nenhuma palavra "negativa".
- `neutral`: misto equilibrado ("produto bom mas atrasou muito"), ou
  puramente factual ("recebi hoje", "ainda não usei").

**categories**: lista separada por `|`, só para **reclamações**. Pode ficar
vazia. A categoria é marcada sempre que o texto relata o problema, *inclusive
em avaliação positiva* ("amei, só demorou um pouco" leva `delivery`). Elogio
não gera categoria.

| chave | quando usar |
|---|---|
| `delivery` | atraso, ainda não chegou, extravio, problema com transportadora/Correios, entregue no lugar errado |
| `product_quality` | defeito, quebrado, danificado no transporte, não funciona, qualidade inferior ao esperado, parece falsificado |
| `wrong_product` | veio outro produto, cor/tamanho/modelo errado, diferente da foto ou da descrição |
| `missing_items` | pedido incompleto: comprou N e veio menos, faltou peça/acessório que deveria vir |
| `customer_service` | sem resposta do vendedor, dificuldade para trocar/devolver/cancelar, reembolso que não sai, cobrança indevida |

Casos específicos:

- "Comprei 2 e só chegou 1" → `missing_items`. Se o texto diz que o outro
  "ainda não chegou", marca também `delivery`.
- "Veio quebrado e ninguém responde" → `product_quality|customer_service`.
- Reclamação apenas de preço ou frete caro → nenhuma categoria (só o aspecto `preco`).
- Reclamação apenas de falta de nota fiscal → `customer_service`, severidade `low`.
- Texto ilegível ou sem relação com a compra ("aaaaa", "teste") → `neutral`,
  sem categoria, com observação em `notes`.

**severity**: `none`, `low`, `medium`, `high`.

- `none`: nenhum problema relatado.
- `low`: incômodo pequeno ou já resolvido; não impede o uso
  ("demorou um pouco", "embalagem simples", "cor levemente diferente").
- `medium`: problema real, mas o cliente tem o produto e consegue usar:
  atraso relevante já entregue, defeito pequeno, tamanho errado,
  atendimento ruim sem prejuízo direto.
- `high`: não recebeu (e passou do prazo), produto quebrado/inutilizável,
  produto totalmente errado, item pago que não veio, cobrança sem entrega,
  menção a Procon/Reclame Aqui/golpe.

Se houver categoria, a severidade não pode ser `none`.

**aspects**: lista separada por `|` dos assuntos mencionados, em elogio ou crítica:
`entrega`, `produto`, `embalagem`, `atendimento`, `preco`, `loja` (a loja ou o
site em geral: "loja confiável", "recomendo o vendedor", "site lannister").

**evidence**: o menor trecho **copiado literalmente** do texto que justifica o
rótulo principal (a reclamação principal; se não houver, o sentimento). Precisa
ser substring exata, porque a avaliação do LLM verifica isso.

**problem**: frase curta, parafraseada, descrevendo o problema. Vazia se não há
problema. Na avaliação do LLM, mede se o modelo detecta *que existe* um problema
(precision/recall de "tem problema"); o texto em si não é comparado, porque
frase livre não tem gabarito único.
