import t2_libs.generallib as gl
import t2_libs.neuralnet as nn
import click

@click.command()
@click.option('-i', prompt = 'Dataset', help = 'Arquivo do Dataset.')
@click.option('-w', prompt = 'Pesos', help = 'Arquivo dos Pesos.')
@click.option('-n', prompt = 'Estrutura da Rede', help = 'Arquivo da Estrutura da Rede.')
@click.option('-m', prompt = 'Modo de teste:', type = click.Choice(['numerica', 'backprop']), help = 'Modo de teste.')
@click.option('-e', default = 0.0000010000, help = 'Valor do ε.')

def main(m, i, w, n, e):
    network_file = n
    weights_file = w
    dataset_file = i  
    estrutura_rede = gl.read_networkstructfile(network_file)
    pesos_iniciais = gl.read_initialweightsfile(weights_file)
    X_train, y_train = gl.read_dataset(dataset_file)
    
    if m == "numerica":
        
        # Rede Neural por meio da verificação numérica
        net = nn.NeuralNet(X_train, 
                           y_train, 
                           hidden_lengths = estrutura_rede["hidden_lengths"], 
                           fator_reg = estrutura_rede["fator_reg"],
                           num_entrada = estrutura_rede["num_saida"],
                           num_saida = estrutura_rede["num_entrada"],
                           initial_weights = pesos_iniciais,
                           numeric = True
                           )
            
        gradients_numeric = net.fit_numeric(e = e)
        string_numeric = net.string_gradients(gradients_numeric, 10)
        
        # Rede Neural por meio do backpropagation
        net = nn.NeuralNet(X_train, 
                           y_train, 
                           hidden_lengths = estrutura_rede["hidden_lengths"], 
                           fator_reg = estrutura_rede["fator_reg"],
                           num_entrada = estrutura_rede["num_saida"],
                           num_saida = estrutura_rede["num_entrada"],
                           initial_weights = pesos_iniciais,
                           numeric = True
                           )
        
        gradients_backpropagation = net.fit_back()
        string_back = net.string_gradients(gradients_backpropagation, 10)
        
        # Calcula erro entre os dois modos
        erro = gl.string_erro_gradients(gradients_numeric, gradients_backpropagation)
        gl.salvar_dados_corretude(string_numeric, string_back, erro)
        
        # Roda rede neural e salva os resultados
        net = nn.NeuralNet(X_train, 
                           y_train, 
                           hidden_lengths = estrutura_rede["hidden_lengths"], 
                           fator_reg = estrutura_rede["fator_reg"],
                           num_entrada = estrutura_rede["num_saida"],
                           num_saida = estrutura_rede["num_entrada"],
                           initial_weights = pesos_iniciais,
                           numeric = True
                           )
        net.fit(save_gradients = True, verbose = False)
      
    elif m == "backprop":
        # Roda rede neural e salva os resultados
        net = nn.NeuralNet(X_train, 
                           y_train, 
                           hidden_lengths = estrutura_rede["hidden_lengths"], 
                           fator_reg = estrutura_rede["fator_reg"],
                           num_entrada = estrutura_rede["num_saida"],
                           num_saida = estrutura_rede["num_entrada"],
                           initial_weights = pesos_iniciais,
                           numeric = True
                           )
        net.fit(save_gradients = True, verbose = False)

  
if __name__ == '__main__':
    main()