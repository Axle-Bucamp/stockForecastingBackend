class GraphforecastPred(nn.Module):
    """
    Graph-based Stock Forecasting Model with Transformer

    Parameters:
        input_shape (tuple[int]): (num_features, sequence_length)
        num_stocks (int): Number of stocks (nodes in the graph)
        dropout (float): Dropout rate for regularization
    """
    def __init__(self, input_shape, dropout: float = 0.2, num_heads=2, num_layers=2):
        super(GraphforecastPred, self).__init__()

        self.num_stocks = 9 # input_shape[0] # num_stocks  
        num_features = 7 # input_shape[2] # input_shape[0]  # **Set to 7 based on indicator list**
        seq_len = 36 # input_shape[1] # input_shape[1]

        # **🔥 Trainable Adjacency Matrix for GCN**
        self.adj_matrix = nn.Parameter(torch.randn(self.num_stocks, self.num_stocks))  

        # **🔥 Graph Convolutional Network (GCN)**
        self.gcn1 = GCNConv(self.num_stocks, 64)  # **Ensure correct input_dim = 7**
        self.gcn2 = GCNConv(64, 128)

        # **Conv1D Feature Extractor**
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(seq_len)  # Ensure output matches seq_len

        # **Transformer for Temporal Modeling**
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(128)

        # **Attention Mechanism**
        self.attn = nn.Linear(128, 1)  
        self.softmax = nn.Softmax(dim=1)

        # **Fully Connected Output**
        self.fc = nn.Linear(128, 1)
        self.activation = nn.LeakyReLU(0.1)
    def forward(self, x):
        """
        Forward pass with GCN and Transformer

        Args:
            x: Tensor of shape (batch, num_stocks, num_features, sequence_length)
        
        Returns:
            Predicted stock values
        """
        
        batch_size, num_stocks, seq_len, num_features = x.shape

        # **🔥 Trainable Adjacency Matrix**
        adj_matrix = torch.softmax(self.adj_matrix, dim=1)  
        edge_index, edge_attr = dense_to_sparse(adj_matrix)

        # **🔥 Ensure Correct Input for GCN**
        
        x_gcn = x.mean(dim=-1)  # **Reduce time dimension (batch, num_stocks, num_features)**
        x_gcn = x_gcn.view(batch_size * seq_len, num_stocks) #num_stocks  # **Flatten for GCN**

        # **🔥 Graph Learning**
        x_gcn = self.gcn1(x_gcn, edge_index).relu()
        x_gcn = self.gcn2(x_gcn, edge_index).relu()

        # **🔥 Conv1D Feature Extraction**
        x = x.view(batch_size * num_stocks, num_features, seq_len)   # (batch * stocks, features, seq_len)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.pool(x)

        # **🔥 Transformer Input Preparation**
        x = x.view(batch_size, num_stocks, 128, seq_len)  # (batch, num_stocks, features, seq_len)
        x = x.mean(dim=1)  # Aggregate over stocks → (batch, 128, seq_len)
        x = x.permute(2, 0, 1)  # (seq_len, batch, features) → Transformer format

          # **🔥 Apply Transformer**
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch, seq_len, 128)
        x_gcn = x_gcn.view(batch_size, seq_len, 128)

        # **🔥 Merge GCN and Transformer Outputs**
        x = x + x_gcn  # Now correctly shaped

        # **🔥 Attention Mechanism**
        attn_weights = self.softmax(self.attn(x))  
        x = torch.sum(x * attn_weights, dim=1)  # Weighted sum over sequence

        # **🔥 Final Prediction**
        x = self.fc(x)
        x = self.activation(x)

        return x