from kedro.pipeline import Node, Pipeline

from .nodes import read_and_prepare_files, combine_dataframes_horizontally, calculate_average_columns, create_transposed_dataframe, create_model_input_table, preprocess_companies, preprocess_shuttles


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=read_and_prepare_files,
                inputs={
                    "directory": "params:data_directory",
                    "file_ids": "params:file_ids",
                    "base_columns": "params:base_columns"
                },
                outputs="prepared_dfs_list",
                name="read_and_prepare_files_node",
            ),
            Node(
                func=combine_dataframes_horizontally,
                inputs="prepared_dfs_list",
                outputs="combined_df",
                name="combine_prepared_dataframes_node",
            ),
            Node(
                func=calculate_average_columns,
                inputs="combined_df",
                outputs="final_df",
                name="calculate_average_columns_node",
            ),
            Node(
                func=create_transposed_dataframe,
                inputs={"df": "final_df", "column_name": "params:cl_e_avg_col_name"},
                outputs="df_e_avg_transposed",
                name="transpose_cl_e_avg_node",
            ),
            Node(
                func=create_transposed_dataframe,
                inputs={"df": "final_df", "column_name": "params:cl_b_avg_col_name"},
                outputs="df_b_avg_transposed",
                name="transpose_cl_b_avg_node",
            ),
            
        ]
    )
